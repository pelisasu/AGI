#!/usr/bin/env python3
"""
=============================================================================
XAUUSD AGI QUANT ENGINE v31.2 (LEVEL 3 - 4 TIER SYSTEM)
=============================================================================
- 10 institutional engines (BOS, FVG, OB, Liquidity, dll)
- Precision entry system (limit order di zona)
- Structural SL/TP
- TIERED LOT SIZING:
    A Super (score >= 90) -> 3.0x (Super Large)
    A+++    (score >= 85) -> 2.0x (Large)
    A++     (score >= 75) -> 1.0x (Normal)
    A       (score >= 65) -> 0.5x (Small)
- Engine Tracker: auto-disable engine jelek per regime
- Auto-Tuner: parameter optimal dari backtest
- AGI layer: regime, memory, calibration, meta
- Gemini debate + reflection
=============================================================================
"""

import os
import sys
import json
import time
import sqlite3
import datetime
from contextlib import closing
import pytz
import numpy as np
import pandas as pd
import requests
import websocket
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agi_core import (
    detect_regime, detect_anomaly, Memory, Calibrator, MetaLearner,
    grade_signal, grade_at_least, compose_local_insight,
)
from agi_gemini import debate, reflect
from telegram_utils import send_text, send_photo, is_paused
from engines import ENGINES
from precision_entry import calculate_precise_entry
from engine_tracker import EngineTracker
from auto_tuner import load_tuned

# =============================================================================
WIB = pytz.timezone("Asia/Jakarta")
UTC = pytz.UTC

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
HEALTHCHECK_URL = os.getenv("HEALTHCHECK_URL", "").strip()

SYMBOL_DERIV = os.getenv("SYMBOL_DERIV", "frxXAUUSD")

_tuned = load_tuned()
_tp = _tuned.get("params", {})

MIN_CONFLUENCE = float(os.getenv("MIN_CONFLUENCE_SCORE",
                                  str(_tp.get("min_confluence", 60))))
MIN_GRADE = os.getenv("MIN_SIGNAL_GRADE", _tp.get("min_grade", "A"))
MIN_PRECISION = float(os.getenv("MIN_PRECISION_SCORE",
                                 str(_tp.get("min_precision", 70))))
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "60"))
FORCE_RUN = os.getenv("FORCE_RUN", "false").lower() == "true"
REQUIRE_MTF_ALIGN = os.getenv("REQUIRE_MTF_ALIGN", "true").lower() == "true"

YAHOO_OFFSET_DEFAULT = float(os.getenv("YAHOO_OFFSET", "-35.00"))
OFFSET_SAFETY_MIN = float(os.getenv("YAHOO_OFFSET_MIN", "-10.00"))
OFFSET_SAFETY_MAX = float(os.getenv("YAHOO_OFFSET_MAX", "-60.00"))

CACHE_DIR = ".state_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
DB_FILE = os.path.join(CACHE_DIR, "state.db")
JOURNAL_FILE = os.path.join(CACHE_DIR, "trade_journal.json")
FAIL_FILE = os.path.join(CACHE_DIR, "failure_count.json")


def log(m):
    print(f"[{datetime.datetime.now(WIB).strftime('%H:%M:%S')}] {m}", flush=True)


def _load(p, d):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return d


def _save(p, d):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, default=str)
    os.replace(tmp, p)


def _ping(status):
    if HEALTHCHECK_URL:
        try:
            requests.get(f"{HEALTHCHECK_URL}?status={status}", timeout=5)
        except Exception:
            pass


# =============================================================================
# PERSISTENCE
# =============================================================================
def _init_db():
    with closing(sqlite3.connect(DB_FILE)) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, direction TEXT, price REAL, zone REAL, status TEXT)""")
        c.commit()


def _price_zone(p, b=5.0):
    return round(p / b) * b


def is_duplicate(direction, price, cooldown):
    zone = _price_zone(price)
    with closing(sqlite3.connect(DB_FILE)) as c:
        rows = c.execute(
            "SELECT ts, direction, price, zone FROM signals "
            "ORDER BY id DESC LIMIT 5").fetchall()
    now = time.time()
    for ts, ld, lp, lz in rows:
        if (now - ts) / 60 < cooldown:
            if direction == ld and abs(zone - lz) < 1e-6:
                return True
            if direction != ld and abs(price - lp) < 2.0:
                return True
    return False


def save_signal(direction, price):
    with closing(sqlite3.connect(DB_FILE)) as c:
        c.execute("INSERT INTO signals (ts, direction, price, zone, status) "
                  "VALUES (?,?,?,?,?)",
                  (time.time(), direction, float(price),
                   _price_zone(price), "ACTIVE"))
        c.commit()


# =============================================================================
# DATA FEED
# =============================================================================
def fetch_deriv(limit=300, gran=900):
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    headers = ["User-Agent: Mozilla/5.0"]
    for attempt in range(3):
        ws = None
        try:
            ws = websocket.create_connection(url, timeout=8, header=headers)
            ws.settimeout(8)
            ws.send(json.dumps({"ticks_history": SYMBOL_DERIV, "count": limit,
                                "end": "latest", "granularity": gran,
                                "style": "candles"}))
            start = time.time()
            while time.time() - start < 8:
                try:
                    res = json.loads(ws.recv())
                except Exception:
                    continue
                if res.get("error"):
                    break
                if "candles" in res and res["candles"]:
                    df = pd.DataFrame(res["candles"])
                    for c in ("close", "high", "low", "open"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df["volume"] = 100.0
                    if "epoch" in df.columns:
                        df.index = pd.to_datetime(df["epoch"], unit="s",
                                                    utc=True)
                    df = df.dropna(subset=["close", "high", "low", "open"])
                    if df.empty:
                        break
                    df = df[["open", "high", "low", "close", "volume"]]
                    return df, float(df["close"].iloc[-1])
        except Exception:
            pass
        finally:
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass
        time.sleep(2)
    return None, None


def fetch_yf_gc(period="5d", interval="15m", min_len=50):
    try:
        raw = yf.Ticker("GC=F").history(period=period, interval=interval)
        if raw is None or len(raw) < min_len:
            return None, None
        price = float(raw["Close"].iloc[-1])
        raw = raw.reset_index()
        dc = next((c for c in raw.columns if "date" in c.lower()),
                  raw.columns[0])
        raw = raw.rename(columns={dc: "datetime", "Close": "close",
                                   "High": "high", "Low": "low",
                                   "Open": "open"})
        if "Volume" in raw.columns:
            raw["volume"] = pd.to_numeric(raw["Volume"],
                                           errors="coerce").fillna(100.0)
        else:
            raw["volume"] = 100.0
        raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True,
                                          errors="coerce")
        raw = raw.dropna(subset=["datetime", "close", "high", "low", "open"])
        raw = raw.set_index("datetime")
        return raw[["open", "high", "low", "close", "volume"]].tail(300), price
    except Exception:
        return None, None


def _clamp_offset(offset):
    if offset > 0:
        offset = -abs(offset)
    return max(OFFSET_SAFETY_MAX, min(OFFSET_SAFETY_MIN, offset))


def get_price_and_data():
    source = "None"
    raw_price = None
    df_m15 = None
    offset = 0.0

    log("Trying Deriv...")
    df_m15, p = fetch_deriv(300, 900)
    if p:
        raw_price = p
        source = "Deriv"
        try:
            offset = float(os.getenv("DERIV_OFFSET", "0"))
        except Exception:
            offset = 0.0

    if raw_price is None:
        log("Trying Yahoo GC=F...")
        df_yf, p = fetch_yf_gc("5d", "15m")
        if p:
            raw_price = p
            source = "Yahoo GC=F"
            df_m15 = df_yf
            offset = _clamp_offset(YAHOO_OFFSET_DEFAULT)
            log(f"[Offset] default={YAHOO_OFFSET_DEFAULT:+.2f} "
                f"clamped={offset:+.2f}")

    if raw_price is None or df_m15 is None or df_m15.empty:
        raise RuntimeError("All feeds failed")

    try:
        offset += float(os.getenv("MT5_OFFSET", "0"))
    except Exception:
        pass

    final = raw_price + offset
    if offset != 0:
        df_m15 = df_m15.copy()
        for c in ("open", "high", "low", "close"):
            df_m15[c] = df_m15[c].astype(float) + offset

    log(f"Source={source} raw={raw_price:.2f} offset={offset:+.2f} "
        f"final={final:.2f}")

    df_h1, _ = fetch_deriv(200, 3600)
    if df_h1 is None:
        df_h1, _ = fetch_yf_gc("60d", "1h", min_len=100)
    if df_h1 is not None and offset != 0:
        df_h1 = df_h1.copy()
        for c in ("open", "high", "low", "close"):
            df_h1[c] = df_h1[c].astype(float) + offset
    if df_h1 is None:
        df_h1 = df_m15

    df_h4, _ = fetch_deriv(200, 14400)
    if df_h4 is None:
        df_h4, _ = fetch_yf_gc("1y", "1d", min_len=50)
    if df_h4 is not None and offset != 0:
        df_h4 = df_h4.copy()
        for c in ("open", "high", "low", "close"):
            df_h4[c] = df_h4[c].astype(float) + offset

    return final, df_m15, df_h1, df_h4, source, offset


# =============================================================================
# INDICATORS
# =============================================================================
def calc_atr(df, p=14):
    try:
        if df is None or len(df) < p + 1:
            return 8.0
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(),
                        (l - c.shift()).abs()], axis=1).max(axis=1)
        v = tr.rolling(p).mean().iloc[-1]
        return float(v) if not pd.isna(v) and v > 0 else 8.0
    except Exception:
        return 8.0


def rsi(df, p=14):
    try:
        if df is None or len(df) < p + 1:
            return pd.Series([50.0])
        d = df["close"].diff()
        g = d.where(d > 0, 0).rolling(p).mean()
        ls = -d.where(d < 0, 0).rolling(p).mean()
        rs = g / (ls + 1e-9)
        return (100 - 100 / (1 + rs)).fillna(50.0)
    except Exception:
        return pd.Series([50.0])


def trend_label(df):
    try:
        s50 = df["close"].rolling(50).mean().iloc[-1]
        pr = float(df["close"].iloc[-1])
        if pd.isna(s50):
            return "NEUTRAL"
        if pr > s50:
            return "BULLISH (UP)"
        if pr < s50:
            return "BEARISH (DOWN)"
        return "SIDEWAYS"
    except Exception:
        return "NEUTRAL"


# =============================================================================
# EVALUATION
# =============================================================================
def _parse_iso(s):
    try:
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = WIB.localize(dt)
        return dt.astimezone(UTC)
    except Exception:
        return None


def evaluate_trade(trade, df_m15, max_hours=6.0):
    entry_dt = _parse_iso(trade.get("time", ""))
    if entry_dt is None:
        return ("INVALID", True)
    if df_m15 is None or df_m15.empty:
        return (None, False)
    if isinstance(df_m15.index, pd.DatetimeIndex):
        sub = df_m15[df_m15.index >= entry_dt]
    else:
        sub = df_m15
    if sub.empty:
        age = (datetime.datetime.now(UTC) - entry_dt).total_seconds() / 3600
        return ("DRAW", True) if age > max_hours else (None, False)

    sig = trade["signal"]
    sl, tp1 = float(trade["sl"]), float(trade["tp1"])
    for _, row in sub.iterrows():
        if sig == "BUY":
            if row["low"] <= sl:
                return ("LOSS", True)
            if row["high"] >= tp1:
                return ("WIN", True)
        else:
            if row["high"] >= sl:
                return ("LOSS", True)
            if row["low"] <= tp1:
                return ("WIN", True)
    age = (datetime.datetime.now(UTC) - entry_dt).total_seconds() / 3600
    return ("DRAW", True) if age > max_hours else (None, False)


def process_evaluated(journal, df_m15, memory, calibrator, meta, tracker):
    changed = False
    for t in journal:
        if t.get("evaluated"):
            continue
        res, ch = evaluate_trade(t, df_m15)
        if not ch:
            continue
        t["evaluated"] = True
        t["result"] = res
        changed = True

        if res in ("WIN", "LOSS"):
            pnl_r = 0.0
            if res == "WIN":
                try:
                    risk = abs(t["price"] - t["sl"])
                    reward = abs(t["tp1"] - t["price"])
                    pnl_r = reward / risk if risk > 0 else 1.0
                except Exception:
                    pnl_r = 1.0
            else:
                pnl_r = -1.0
            t["pnl_r"] = pnl_r

            mid = t.get("memory_id")
            if mid:
                memory.update(mid, res, pnl_r)

            regime = t.get("regime", "TRANSITION")
            meta.update(regime, res == "WIN", pnl_r)

            engine_states = t.get("engine_states", {})
            for name, st in engine_states.items():
                sc = st.get("sc", 0)
                tracker.update(regime, name, sc, t["signal"], res)

            conf = t.get("adjusted_consensus", t.get("consensus", 60)) / 100.0
            calibrator.accumulate(conf, 1 if res == "WIN" else 0)

    if changed:
        _save(JOURNAL_FILE, journal)
    return changed


# =============================================================================
# CHART
# =============================================================================
def make_chart(df, entry_data, signal, conf, atr):
    try:
        plt.figure(figsize=(11, 6))
        sub = df.tail(100)
        x = sub.index if isinstance(sub.index, pd.DatetimeIndex) \
            else range(len(sub))
        plt.plot(x, sub["close"].values, color="gold", linewidth=1.5,
                 label="M15")

        plt.axhspan(entry_data.entry_low, entry_data.entry_high,
                    color="cyan", alpha=0.25,
                    label=f"Entry {entry_data.entry_low:.2f}-"
                          f"{entry_data.entry_high:.2f}")
        plt.axhline(entry_data.entry_ideal, color="cyan",
                    linestyle="--", linewidth=1)
        plt.axhline(entry_data.sl, color="red", linewidth=2,
                    label=f"SL {entry_data.sl:.2f}")
        plt.axhline(entry_data.tp1, color="green", linestyle=":",
                    label=f"TP1 {entry_data.tp1:.2f} ({entry_data.rr_tp1}R)")
        plt.axhline(entry_data.tp2, color="green", linestyle="--",
                    label=f"TP2 {entry_data.tp2:.2f} ({entry_data.rr_tp2}R)")
        plt.axhline(entry_data.tp3, color="green", linestyle="-",
                    label=f"TP3 {entry_data.tp3:.2f} ({entry_data.rr_tp3}R)")

        plt.title(f"{signal} | conf {conf:.0f}% | "
                  f"Precision {entry_data.precision_score:.0f} "
                  f"({entry_data.precision_grade}) | "
                  f"{entry_data.entry_type}")
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(CACHE_DIR, "chart.png")
        plt.savefig(p, dpi=140)
        plt.close("all")
        return p
    except Exception as e:
        log(f"chart err {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# =============================================================================
# CAPTION WITH 4-TIER LOT SIZING
# =============================================================================
def compose_precision_caption(signal, entry_data, atr, consensus,
                               grade, grade_score, regime, anomaly,
                               mem_stats, debate_res, source, offset,
                               ai_insight):
    import html as _html

    # =========================================================
    # 4-TIER LOT SIZING
    # =========================================================
    LOT_TIERS = {
        "A Super": {"mult": 3.0, "label": "🚀 SUPER", "risk": "3.0%",
                    "note": "Exceptional setup - maximum size"},
        "A+++":    {"mult": 2.0, "label": "🔴 LARGE", "risk": "2.0%",
                    "note": "High conviction - full size"},
        "A++":     {"mult": 1.0, "label": "🟡 NORMAL", "risk": "1.0%",
                    "note": "Standard setup"},
        "A":       {"mult": 0.5, "label": "🟢 SMALL", "risk": "0.5%",
                    "note": "Probing - reduce size"},
        "B":       {"mult": 0.0, "label": "⚫ SKIP", "risk": "0%",
                    "note": "Below threshold"},
    }
    tier = LOT_TIERS.get(grade, LOT_TIERS["A"])

    bar_f = int(consensus / 10)
    bar = "█" * bar_f + "░" * (10 - bar_f)
    regime_tag = {
        "TRENDING_UP": "📈 TREND UP", "TRENDING_DOWN": "📉 TREND DOWN",
        "RANGING": "↔️ RANGING", "VOLATILE": "⚡ VOLATILE",
        "QUIET": "😴 QUIET", "TRANSITION": "🔄 TRANSITION",
    }.get(regime.regime.value, "❓")
    emoji = "🟢" if signal == "BUY" else "🔴"

    # Highlight untuk A Super
    super_banner = ""
    if grade == "A Super":
        super_banner = "\n🔥🔥🔥 <b>EXCEPTIONAL SIGNAL</b> 🔥🔥🔥"

    mem_line = "—"
    if mem_stats.get("n", 0) >= 5:
        mem_line = (f"{mem_stats['n']} case | WR {mem_stats['winrate']:.0f}% "
                    f"| avg {mem_stats['avg_r']:+.2f}R")

    debate_line = ""
    if debate_res.get("verdict") and debate_res["verdict"] != "ABSTAIN":
        de = {"AGREE": "✅", "DISAGREE": "❌"}.get(debate_res["verdict"], "⚪")
        debate_line = f"\n{de} <b>AI Council:</b> {debate_res['verdict']}"
        if debate_res.get("notes"):
            debate_line += (f" — <i>"
                            f"{_html.escape(debate_res['notes'][:100])}</i>")

    tp_lines = "\n".join([
        f"  <b>TP{i+1}:</b> <code>{tp:.2f}</code> "
        f"({rr:.2f}R) <i>{entry_data.tp_reasons[i]}</i>"
        for i, (tp, rr) in enumerate([
            (entry_data.tp1, entry_data.rr_tp1),
            (entry_data.tp2, entry_data.rr_tp2),
            (entry_data.tp3, entry_data.rr_tp3),
            (entry_data.tp4, 0.0),
        ])
    ])

    off_line = f" | offset {offset:+.2f}" if offset else ""
    ai_safe = _html.escape(ai_insight[:350]) if ai_insight else "-"

    entry_type_label = {
        "FVG_FILL": "🎯 FVG Fill",
        "OB_RETEST": "🎯 Order Block Retest",
        "STRUCTURE_PULLBACK": "🎯 Structure Pullback",
    }.get(entry_data.entry_type, "🎯")

    return (
        f"{emoji} <b>XAUUSD {signal}</b> — Grade <b>{grade}</b>"
        f"{super_banner}\n"
        f"<b>Precision: {entry_data.precision_score:.0f}/100 "
        f"({entry_data.precision_grade})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>LOT TIER: {tier['label']}</b> ({tier['mult']}x)\n"
        f"⚠️ <b>Risk per trade:</b> {tier['risk']} equity\n"
        f"📝 <i>{tier['note']}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Confidence: [{bar}] {consensus:.0f}%\n"
        f"🌊 Regime: {regime_tag} (conf {regime.confidence:.2f})\n"
        f"📡 Source: {_html.escape(source)}{off_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{entry_type_label}\n"
        f"<b>Entry Zone:</b> <code>{entry_data.entry_low:.2f} - "
        f"{entry_data.entry_high:.2f}</code>\n"
        f"<b>Entry Ideal:</b> <code>{entry_data.entry_ideal:.2f}</code>\n"
        f"<b>SL:</b> <code>{entry_data.sl:.2f}</code> "
        f"<i>({entry_data.sl_reason})</i>\n"
        f"<b>Risk:</b> {entry_data.risk_points:.2f} pts\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{tp_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 Memory: {mem_line}\n"
        f"⚡ Anomaly: {anomaly.score:.2f}"
        f"{debate_line}\n"
        f"🤖 <i>{ai_safe}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {datetime.datetime.now(WIB).strftime('%H:%M WIB | %d-%m')}"
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 60)
    log("XAUUSD AGI ENGINE v31.2 (4-TIER) START")
    log("=" * 60)
    _ping("start")

    _init_db()
    journal = _load(JOURNAL_FILE, [])
    fail = _load(FAIL_FILE, {"count": 0})

    if is_paused() and not FORCE_RUN:
        log("Paused. Skip.")
        _ping("paused")
        return 0

    if GEMINI_API_KEY:
        log(f"[Gemini] API key present (len={len(GEMINI_API_KEY)})")
    else:
        log("[Gemini] ⚠️ No API key")

    if _tp:
        log(f"[L3] Tuned params loaded: {_tp}")

    try:
        price, df_m15, df_h1, df_h4, source, offset = get_price_and_data()
        fail["count"] = 0
        _save(FAIL_FILE, fail)
    except Exception as e:
        fail["count"] = fail.get("count", 0) + 1
        _save(FAIL_FILE, fail)
        log(f"Data fetch failed: {e}")
        if fail["count"] >= 2:
            send_text(TELEGRAM_BOT_TOKEN,
                      f"🚨 <b>ALERT</b>: fetch gagal {fail['count']}x", None)
        _ping("fail_fetch")
        return 1

    memory = Memory()
    calibrator = Calibrator()
    meta = MetaLearner()
    tracker = EngineTracker()

    process_evaluated(journal, df_m15, memory, calibrator, meta, tracker)

    regime = detect_regime(df_m15)
    anomaly = detect_anomaly(df_m15)
    _, mem_stats = memory.query(df_m15, k=20)
    current_regime = regime.regime.value

    h1_trend = trend_label(df_h1)
    h4_trend = trend_label(df_h4) if df_h4 is not None else "NEUTRAL"
    h1_rsi = float(rsi(df_h1).iloc[-1])

    buy_w = sell_w = 0.0
    states = {}
    bullish_engines = bearish_engines = 0
    disabled_engines = []

    for name, fn in ENGINES:
        try:
            sc, w = fn(df_m15)
            tracker_mult = tracker.get_weight_mult(current_regime, name)
            if tracker_mult == 0.0:
                disabled_engines.append(name)
            effective_w = w * tracker_mult
            states[name] = {
                "sc": sc,
                "weight_used": round(effective_w, 4),
                "tracker_mult": round(tracker_mult, 3),
            }
            if sc > 0:
                buy_w += effective_w * abs(sc)
                bullish_engines += 1
            elif sc < 0:
                sell_w += effective_w * abs(sc)
                bearish_engines += 1
        except Exception as e:
            states[name] = {"sc": 0, "weight_used": 1.0, "tracker_mult": 1.0}
            log(f"Engine {name} err: {e}")

    if disabled_engines:
        log(f"[L3] Disabled in {current_regime}: "
            f"{', '.join(disabled_engines)}")

    if "BULLISH" in h1_trend:
        buy_w += 2.5
    elif "BEARISH" in h1_trend:
        sell_w += 2.5

    if df_h4 is not None:
        if "BULLISH" in h4_trend:
            buy_w += 1.0
        elif "BEARISH" in h4_trend:
            sell_w += 1.0

    total_w = buy_w + sell_w
    if total_w == 0 or buy_w == sell_w:
        log("Neutral. Skip.")
        _ping("neutral")
        return 0

    consensus = max(buy_w, sell_w) / total_w * 100
    signal = "BUY" if buy_w > sell_w else "SELL"

    penalty = meta.penalty(current_regime)
    adjusted = max(0, min(100, consensus * penalty))

    log(f"Signal={signal} consensus={consensus:.1f}% "
        f"adjusted={adjusted:.1f}% regime={current_regime} "
        f"anomaly={anomaly.score:.2f} bull={bullish_engines} "
        f"bear={bearish_engines}")

    if adjusted < MIN_CONFLUENCE:
        log(f"Below threshold {MIN_CONFLUENCE}. Skip.")
        _ping("low_score")
        return 0

    if REQUIRE_MTF_ALIGN and not FORCE_RUN:
        aligned = (
            (signal == "BUY" and "BULLISH" in h1_trend and
             ("BULLISH" in h4_trend or h4_trend == "NEUTRAL")) or
            (signal == "SELL" and "BEARISH" in h1_trend and
             ("BEARISH" in h4_trend or h4_trend == "NEUTRAL"))
        )
        if not aligned:
            log(f"MTF not aligned: {signal} H1={h1_trend} H4={h4_trend}")
            _ping("mtf_unaligned")
            return 0

    if anomaly.is_anomaly and not FORCE_RUN:
        log(f"Anomaly {anomaly.score:.2f}. Skip.")
        _ping("anomaly")
        return 0

    atr_val = calc_atr(df_m15)
    grade_info = grade_signal(adjusted, signal, states, h1_trend, h4_trend,
                               atr_val, mem_stats, anomaly.score)
    log(f"Grade={grade_info['grade']} score={grade_info['score']}")

    if not grade_at_least(grade_info["grade"], MIN_GRADE) and not FORCE_RUN:
        log(f"Grade below {MIN_GRADE}. Skip.")
        _ping("low_grade")
        return 0

    # =========================================================================
    # [NEW] A SUPER UPGRADE
    # =========================================================================
    # Kalau Grade A+++ DAN score >= 90 -> upgrade ke A Super (3x lot)
    if grade_info["grade"] == "A+++" and grade_info["score"] >= 90:
        original_grade = grade_info["grade"]
        original_score = grade_info["score"]
        grade_info["grade"] = "A Super"
        log(f"🔥 UPGRADE: {original_grade} (score {original_score}) "
            f"-> A Super (max lot 3x)")

    log("Calculating precision entry...")
    entry_data = calculate_precise_entry(
        df_m15, signal, price, h4_trend, h1_trend, adjusted, atr_val
    )
    log(f"[Precision] {entry_data.precision_grade} "
        f"({entry_data.precision_score}/100) "
        f"type={entry_data.entry_type} "
        f"entry={entry_data.entry_low:.2f}-{entry_data.entry_high:.2f} "
        f"SL={entry_data.sl:.2f} "
        f"TP1={entry_data.tp1:.2f} ({entry_data.rr_tp1}R)")

    if entry_data.precision_score < MIN_PRECISION and not FORCE_RUN:
        log(f"Precision {entry_data.precision_score} < {MIN_PRECISION}. Skip.")
        _ping("low_precision")
        return 0

    if is_duplicate(signal, price, COOLDOWN_MIN) and not FORCE_RUN:
        log("Duplicate. Skip.")
        _ping("duplicate")
        return 0

    if GEMINI_API_KEY:
        debate_res = debate(signal, price, adjusted, current_regime,
                            h1_trend, h4_trend, h1_rsi, atr_val,
                            mem_stats, anomaly.score, GEMINI_API_KEY)
        log(f"Debate={debate_res['verdict']} "
            f"mult={debate_res['confidence_mult']}")
    else:
        debate_res = {"verdict": "ABSTAIN", "confidence_mult": 1.0,
                      "notes": "no_key", "raw": ""}

    if debate_res["verdict"] == "DISAGREE" and not FORCE_RUN:
        log("Debate DISAGREE. Skip.")
        _ping("debate_disagree")
        return 0

    mem_id = memory.store(df_m15, regime, signal,
                           entry_data.entry_ideal, outcome="OPEN",
                           extra={"grade": grade_info["grade"],
                                  "precision": entry_data.precision_score})
    trade = {
        "time": datetime.datetime.now(WIB).isoformat(),
        "signal": signal,
        "price": entry_data.entry_ideal,
        "entry_low": entry_data.entry_low,
        "entry_high": entry_data.entry_high,
        "entry_type": entry_data.entry_type,
        "sl": entry_data.sl,
        "tp1": entry_data.tp1,
        "tp2": entry_data.tp2,
        "tp3": entry_data.tp3,
        "tp4": entry_data.tp4,
        "rr_tp1": entry_data.rr_tp1,
        "rr_tp2": entry_data.rr_tp2,
        "rr_tp3": entry_data.rr_tp3,
        "risk_points": entry_data.risk_points,
        "consensus": round(consensus, 2),
        "adjusted_consensus": round(adjusted, 2),
        "grade": grade_info["grade"],
        "grade_score": grade_info["score"],
        "precision_score": entry_data.precision_score,
        "precision_grade": entry_data.precision_grade,
        "regime": current_regime,
        "anomaly_score": anomaly.score,
        "debate_verdict": debate_res["verdict"],
        "memory_winrate": mem_stats.get("winrate", 50),
        "memory_n": mem_stats.get("n", 0),
        "engine_states": states,
        "memory_id": mem_id,
        "source": source,
        "offset_used": offset,
        "evaluated": False,
    }
    journal.append(trade)
    if len(journal) > 200:
        journal = journal[-200:]
    _save(JOURNAL_FILE, journal)
    save_signal(signal, entry_data.entry_ideal)

    ai = compose_local_insight(signal, adjusted, current_regime,
                                h1_trend, h4_trend, mem_stats, anomaly.score)

    caption = compose_precision_caption(
        signal, entry_data, atr_val, adjusted,
        grade_info["grade"], grade_info["score"], regime, anomaly,
        mem_stats, debate_res, source, offset, ai,
    )

    if disabled_engines:
        caption += (f"\n🔕 <i>Disabled di {current_regime}: "
                    f"{', '.join(disabled_engines)}</i>")

    if source == "Yahoo GC=F":
        caption += (f"\n⚠️ <i>Yahoo delay ~10m. "
                    f"Verify harga di MT5 sebelum limit order.</i>")

    chart = make_chart(df_m15, entry_data, signal, adjusted, atr_val)
    if chart and os.path.exists(chart):
        send_photo(TELEGRAM_BOT_TOKEN, caption, chart)
    else:
        send_text(TELEGRAM_BOT_TOKEN, caption, None)

    log(f"✅ SENT: {signal} entry={entry_data.entry_ideal:.2f} "
        f"SL={entry_data.sl:.2f} grade={grade_info['grade']} "
        f"precision={entry_data.precision_grade}")

    reflect(journal, current_regime, GEMINI_API_KEY, interval=10)
    _ping("success")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
