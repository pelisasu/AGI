#!/usr/bin/env python3
"""Telegram helpers: send, broadcast, pause state, bot commands."""

import os
import json
import time
import html
import datetime
import threading
from typing import List, Dict, Optional, Callable
import requests

TG_API = "https://api.telegram.org/bot{token}/{method}"
STATE_FILE = ".state_cache/bot_state.json"
CHAT_REGISTRY = ".state_cache/chats.json"


def _ensure_state_dir():
    os.makedirs(".state_cache", exist_ok=True)


def _load(p: str, default):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def _save_atomic(p: str, data):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, p)


# =============================================================================
# STATE (pause / resume)
# =============================================================================
class BotState:
    def __init__(self, path: str = STATE_FILE):
        self.path = path
        self.data = {"paused": False, "paused_until": None,
                     "last_update_id": 0, "last_signal": None}
        self.data.update(_load(path, {}))

    def save(self):
        _save_atomic(self.path, self.data)

    def is_paused(self) -> bool:
        if not self.data.get("paused"):
            return False
        until = self.data.get("paused_until")
        if until:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            if now > until:
                self.data["paused"] = False
                self.data["paused_until"] = None
                self.save()
                return False
        return True

    def pause(self, hours: Optional[float] = None):
        self.data["paused"] = True
        if hours:
            until = datetime.datetime.now(datetime.timezone.utc) + \
                    datetime.timedelta(hours=hours)
            self.data["paused_until"] = until.timestamp()
        else:
            self.data["paused_until"] = None
        self.save()

    def resume(self):
        self.data["paused"] = False
        self.data["paused_until"] = None
        self.save()


# =============================================================================
# MULTI-CHAT
# =============================================================================
def _parse_chat_ids(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else []


def _load_registry() -> Dict:
    return _load(CHAT_REGISTRY, {})


def _save_registry(reg: Dict):
    _save_atomic(CHAT_REGISTRY, reg)


def get_all_chat_ids() -> List[str]:
    env_ids = _parse_chat_ids(os.getenv("TELEGRAM_CHAT_ID", "").strip())
    reg = _load_registry()
    combined = list(dict.fromkeys(env_ids + list(reg.keys())))
    return combined


def _parse_silent(chat_id: str):
    if chat_id.startswith("@silent:"):
        return chat_id[len("@silent:"):], True
    if chat_id.startswith("silent:"):
        return chat_id[len("silent:"):], True
    return chat_id, False


def add_chat(chat_id: str, tag: str = "user") -> bool:
    reg = _load_registry()
    reg[chat_id] = {"tag": tag, "added_at": time.time(), "active": True}
    _save_registry(reg)
    return True


def remove_chat(chat_id: str) -> bool:
    reg = _load_registry()
    if chat_id in reg:
        del reg[chat_id]
        _save_registry(reg)
        return True
    return False


def list_chats() -> str:
    reg = _load_registry()
    if not reg:
        return "No registered chats."
    lines = ["📋 <b>Registered chats:</b>"]
    for cid, info in reg.items():
        lines.append(f"  • <code>{cid}</code> [{info.get('tag', '-')}]")
    return "\n".join(lines)


# =============================================================================
# SEND
# =============================================================================
def send_text(token: str, text: str, chat_id: Optional[str] = None,
              parse_mode: str = "HTML") -> bool:
    if not token:
        return False
    targets = [chat_id] if chat_id else get_all_chat_ids()
    if not targets:
        return False
    if len(text) > 4000:
        text = text[:3997] + "..."
    ok = True
    for cid in targets:
        actual, silent = _parse_silent(cid)
        try:
            r = requests.post(
                TG_API.format(token=token, method="sendMessage"),
                json={"chat_id": actual, "text": text,
                      "parse_mode": parse_mode,
                      "disable_web_page_preview": True,
                      "disable_notification": silent},
                timeout=15,
            )
            ok = ok and (r.status_code == 200)
        except Exception:
            ok = False
        time.sleep(0.05)
    return ok


def send_photo(token: str, caption: str, photo_path: str,
               chat_id: Optional[str] = None) -> bool:
    if not token or not os.path.exists(photo_path):
        return False
    targets = [chat_id] if chat_id else get_all_chat_ids()
    if not targets:
        return False
    if len(caption) > 1000:
        caption = caption[:997] + "..."
    ok = True
    for cid in targets:
        actual, silent = _parse_silent(cid)
        try:
            with open(photo_path, "rb") as f:
                r = requests.post(
                    TG_API.format(token=token, method="sendPhoto"),
                    data={"chat_id": actual, "caption": caption,
                          "parse_mode": "HTML",
                          "disable_notification": silent},
                    files={"photo": f}, timeout=25,
                )
            ok = ok and (r.status_code == 200)
        except Exception:
            ok = False
        time.sleep(0.05)
    return ok


# =============================================================================
# SIMPLE POLLING BOT (command handler)
# =============================================================================
class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = str(chat_id).split(",")[0].strip()  # owner chat
        self.state = BotState()
        self.handlers: Dict[str, Callable] = {}

    def register(self, cmd: str, handler: Callable):
        self.handlers[cmd.lower()] = handler

    def _send(self, text: str):
        if len(text) > 4000:
            text = text[:3997] + "..."
        try:
            requests.post(
                TG_API.format(token=self.token, method="sendMessage"),
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": "HTML",
                      "disable_web_page_preview": True},
                timeout=15,
            )
        except Exception:
            pass

    def _handle(self, text: str):
        text = (text or "").strip()
        if not text.startswith("/"):
            return
        cmd = text.split()[0][1:].split("@")[0].lower()
        h = self.handlers.get(cmd)
        if h:
            try:
                reply = h(text)
                if reply:
                    self._send(reply)
            except Exception as e:
                self._send(f"⚠️ Error: <code>{html.escape(str(e))}</code>")

    def poll_forever(self, stop_event: Optional[threading.Event] = None):
        while not (stop_event and stop_event.is_set()):
            try:
                offset = self.state.data.get("last_update_id", 0) + 1
                r = requests.get(
                    TG_API.format(token=self.token, method="getUpdates"),
                    params={"offset": offset, "timeout": 25},
                    timeout=30,
                )
                if r.status_code != 200:
                    time.sleep(3)
                    continue
                for upd in r.json().get("result", []):
                    self.state.data["last_update_id"] = upd["update_id"]
                    msg = upd.get("message") or {}
                    if str((msg.get("chat") or {}).get("id")) != self.chat_id:
                        continue
                    self._handle(msg.get("text", ""))
                self.state.save()
            except Exception:
                time.sleep(5)

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.poll_forever, daemon=True)
        t.start()
        return t


def is_paused() -> bool:
    return BotState().is_paused()
