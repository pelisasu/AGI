#!/usr/bin/env python3
"""
=============================================================================
XAUUSD AGI QUANT ENGINE v30.2 (FINAL)
=============================================================================
Cuma butuh 3 secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY.
=================================================================
PATCH TERBARU:
 [v30.2] Offset YF default -35.00 (data aktual 29 Sep 2026)
         Range clamp -10 s/d -60 (contango wajar)
 [v30.2] QUIET regime guard: skip trend signal di pasar ranging lemah
 [v30.2] Yahoo delay warning di caption + journal
 [v30.2] Gemini debug logging (biar ketahuan kalau API key gagal)
 [v30.2] Simpan source + offset_used di journal untuk audit trail
=============================================================================
"""

import os
import sys
import json
import time
import sqlite3
import datetime
from contextlib import closing
from datetime import timedelta
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
    grade_signal, grade_at_least, compose_caption, compose_local_insight,
)
from agi_gemini import debate, reflect, get_last_reflection
from telegram_utils import send_text, send_photo, is_paused

# =============================================================================
# CONFIG
# =============================================================================
WIB = pytz.timezone("Asia/Jakarta")
UTC = pytz.UTC

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
HEALTHCHECK_URL = os.getenv("HEALTHCHECK_URL", "").strip()

SYMBOL_DERIV = os.getenv("SYMBOL_DERIV", "frxXAUUSD")
MIN_CONFLUENCE = float(os.getenv("MIN_CONFLUENCE_SCORE", "60"))
MIN_GRADE = os.getenv("MIN_SIGNAL_GRADE", "A")
COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "60"))
FORCE_RUN = os.getenv("FORCE_RUN", "false").lower() == "true"

# -----------------------------------------------------------------------------
# OFFSET (YF GC=F -> XAUUSD Spot)
# -----------------------------------------------------------------------------
# Data 29 Sep 2026: GC=F 4159.20, Spot ~4124.20 -> spread ~35.00
# Range historis wajar: 10 (tipis) s/d 60 (contango lebar)
YAHOO_OFFSET_DEFAULT = float(os.getenv("YAHOO_OFFSET", "-35.00"))
OFFSET_SAFETY_MIN = float(os.getenv("YAHOO_OFFSET_MIN", "-10.00"))   # batas atas (paling kecil)
OFFSET_SAFETY_MAX = float(os.getenv("YAHOO_OFFSET_MAX", "-60.00"))   # batas bawah (paling besar)

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


def _ping(status: str):
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
                        df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
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
        dc = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])
        raw = raw.rename(columns={dc: "datetime", "Close": "close",
                                   "High": "high", "Low": "low", "Open": "open"})
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


def _clamp_offset(offset: float) -> float:
    """Offset harus negatif (YF > spot) dan di dalam range wajar."""
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
            log(f"[Offset] YF default={YAHOO_OFFSET_DEFAULT:+.2f} "
                f"clamped={offset:+.2f} "
                f"(range {OFFSET_SAFETY_MIN:+.2f}..{OFFSET_SAFETY_MAX:+.2f})")

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

    # H1
    df_h1, _ = fetch_deriv(200, 3600)
    if df_h1 is None:
        df_h1, _ = fetch_yf_gc("60d", "1h", min_len=100)
    if df_h1 is not None and offset != 0:
        df_h1 = df_h1.copy()
        for c in ("open", "high", "low", "close"):
            df_h1[c] = df_h1[c].astype(float) + offset
    if df_h1 is None:
        df_h1 = df_m15

    # H4
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
# 10 ENGINES
# =============================================================================
def NADI(df):
    try:
        e9 = df["close"].ewm(9).mean().iloc[-1]
        e21 = df["close"].ewm(21).mean().iloc[-1]
        p = float(df["close"].iloc[-1])
        if p > e9 and e9 > e21:
            return (1, 1.5)
        if p < e9 and e9 < e21:
            return (-1, 1.5)
        return (0, 1.5)
    except Exception:
        return (0, 1.5)


def SAWAH(df):
    try:
        hi = df["high"].rolling(50).max().iloc[-1]
        lo = df["low"].rolling(50).min().iloc[-1]
        p = float(df["close"].iloc[-1])
        if p > lo + (hi - lo) * 0.618:
            return (1, 1.5)
        if p < lo + (hi - lo) * 0.382:
            return (-1, 1.5)
        return (0, 1.5)
    except Exception:
        return (0, 1.5)


def SEMUT(df):
    try:
        e50 = df["close"].ewm(50).mean().iloc[-1]
        p = float(df["close"].iloc[-1])
        return (1, 1.2) if p > e50 else (-1, 1.2)
    except Exception:
        return (0, 1.2)


def PADI(df):
    try:
        r = float(rsi(df).iloc[-1])
        if r > 55:
            return (1, 1.3)
        if r < 45:
            return (-1, 1.3)
        return (0, 1.3)
    except Exception:
        return (0, 1.3)


def AKAR(df):
    try:
        s = df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 \
            else df["close"].mean()
        p = float(df["close"].iloc[-1])
        return (1, 1.8) if p > s else (-1, 1.8)
    except Exception:
        return (0, 1.8)


def WAYANG(df):
    try:
        hi = df["high"].rolling(20).max().iloc[-1]
        lo = df["low"].rolling(20).min().iloc[-1]
        p = float(df["close"].iloc[-1])
        return (1, 1.0) if p > (hi + lo) / 2 else (-1, 1.0)
    except Exception:
        return (0, 1.0)


def LUMPUR(df):
    try:
        v = float(df["volume"].iloc[-1])
        vm = float(df["volume"].rolling(20).mean().iloc[-1])
        p = float(df["close"].iloc[-1])
        pp = float(df["close"].iloc[-2])
        if v > vm and p > pp:
            return (1, 1.1)
        if v > vm and p < pp:
            return (-1, 1.1)
        return (0, 1.1)
    except Exception:
        return (0, 1.1)


def API(df):
    try:
        b = float((df["close"] - df["open"]).abs().iloc[-1])
        ab = float((df["close"] - df["open"]).abs().rolling(20).mean().iloc[-1])
        p = float(df["close"].iloc[-1])
        pp = float(df["close"].iloc[-2])
        if b > ab and p > pp:
            return (1, 1.0)
        if b > ab and p < pp:
            return (-1, 1.0)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def ANGIN(df):
    try:
        hi = float(df["high"].iloc[-1])
        lo = float(df["low"].iloc[-1])
        c = float(df["close"].iloc[-1])
        if (hi - c) > (c - lo) * 1.5:
            return (-1, 1.0)
        if (c - lo) > (hi - c) * 1.5:
            return (1, 1.0)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def EMBER(df):
    try:
        p = float(df["close"].iloc[-1])
        m = float(df["close"].rolling(10).mean().iloc[-1])
        return (1, 1.0) if p > m else (-1, 1.0)
    except Exception:
        return (0, 1.0)


ENGINES = [("NADI", NADI), ("SAWAH", SAWAH), ("SEMUT", SEMUT),
           ("PADI", PADI), ("AKAR", AKAR), ("WAYANG", WAYANG),
           ("LUMPUR", LUMPUR), ("API", API), ("ANGIN", ANGIN),
           ("EMBER", EMBER)]


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


def process_evaluated(journal, df_m15, memory, calibrator, meta):
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

            conf = t.get("adjusted_consensus", t.get("consensus", 60)) / 100.0
            calibrator.accumulate(conf, 1 if res == "WIN" else 0)

    if changed:
        _save(JOURNAL_FILE, journal)
    return changed


# =============================================================================
# CHART
# =============================================================================
def make_chart(df, entry, sl, tps, signal, conf, atr):
    try:
        plt.figure(figsize=(10, 6))
        sub = df.tail(80)
        x = sub.index if isinstance(sub.index, pd.DatetimeIndex) \
            else range(len(sub))
        plt.plot(x, sub["close"].values, color="gold", linewidth=1.5,
                 label="M15")
        plt.axhline(entry, color="cyan", label=f"Entry {entry:.2f}")
        plt.axhline(sl, color="red", label=f"SL {sl:.2f}")
        plt.axhline(tps[0], color="green", linestyle=":", label="TP1")
        plt.axhline(tps[2], color="green", linestyle="-", label="TP3")
        plt.title(f"{signal} | conf {conf:.0f}% | ATR {atr:.2f}")
        plt.legend(fontsize=8)
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
# MAIN
# =============================================================================
def main():
    log("=" * 60)
    log("XAUUSD AGI ENGINE v30.2 START")
    log("=" * 60)

    _ping("start")

    _init_db()
    journal = _load(JOURNAL_FILE, [])
    fail = _load(FAIL_FILE, {"count": 0})

    if is_paused() and not FORCE_RUN:
        log("Paused via /pause. Skip.")
        _ping("paused")
        return 0

    # Gemini API key check
    if GEMINI_API_KEY:
        log(f"[Gemini] API key present (len={len(GEMINI_API_KEY)})")
    else:
        log("[Gemini] ⚠️ API key MISSING — debate/reflection disabled")

    # Fetch data
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
                      f"🚨 <b>ALERT</b>: data fetch gagal {fail['count']}x",
                      None)
        _ping("fail_fetch")
        return 1

    memory = Memory()
    calibrator = Calibrator()
    meta = MetaLearner()

    process_evaluated(journal, df_m15, memory, calibrator, meta)

    regime = detect_regime(df_m15)
    anomaly = detect_anomaly(df_m15)
    _, mem_stats = memory.query(df_m15, k=20)

    h1_trend = trend_label(df_h1)
    h4_trend = trend_label(df_h4) if df_h4 is not None else "NEUTRAL"
    h1_rsi = float(rsi(df_h1).iloc[-1])

    # Scoring
    buy_w = sell_w = 0.0
    states = {}
    for name, fn in ENGINES:
        try:
            sc, w = fn(df_m15)
            states[name] = {"sc": sc, "weight_used": w}
            if sc > 0:
                buy_w += w * abs(sc)
            elif sc < 0:
                sell_w += w * abs(sc)
        except Exception:
            states[name] = {"sc": 0, "weight_used": 1.0}

    if "BULLISH" in h1_trend:
        buy_w += 2.0
    elif "BEARISH" in h1_trend:
        sell_w += 2.0

    if df_h4 is not None:
        if "BULLISH" in h4_trend:
            buy_w += 0.75
        elif "BEARISH" in h4_trend:
            sell_w += 0.75

    total_w = buy_w + sell_w
    if total_w == 0 or buy_w == sell_w:
        log("Neutral market. Skip.")
        _ping("neutral")
        return 0

    consensus = max(buy_w, sell_w) / total_w * 100
    signal = "BUY" if buy_w > sell_w else "SELL"

    penalty = meta.penalty(regime.regime.value)
    adjusted = consensus * penalty
    adjusted = max(0, min(100, adjusted))

    log(f"Signal={signal} consensus={consensus:.1f}% "
        f"adjusted={adjusted:.1f}% regime={regime.regime.value} "
        f"anomaly={anomaly.score:.2f}")

    if adjusted < MIN_CONFLUENCE:
        log(f"Below threshold {MIN_CONFLUENCE}. Skip.")
        _ping("low_score")
        return 0

    grade_info = grade_signal(adjusted, signal, states, h1_trend, h4_trend,
                               calc_atr(df_m15), mem_stats, anomaly.score)
    log(f"Grade={grade_info['grade']} score={grade_info['score']}")

    # QUIET regime guard
    if (regime.regime.value == "QUIET"
            and grade_info["grade"] in ("A", "B", "C", "D")
            and not FORCE_RUN):
        log(f"QUIET regime + grade {grade_info['grade']} -> skip")
        _ping("quiet_low_grade")
        return 0

    if not grade_at_least(grade_info["grade"], MIN_GRADE) and not FORCE_RUN:
        log(f"Grade below {MIN_GRADE}. Skip.")
        _ping("low_grade")
        return 0

    if anomaly.is_anomaly and not FORCE_RUN:
        log(f"Anomaly {anomaly.score:.2f} too high. Skip.")
        _ping("anomaly_high")
        return 0

    if is_duplicate(signal, price, COOLDOWN_MIN) and not FORCE_RUN:
        log("Duplicate signal. Skip.")
        _ping("duplicate")
        return 0

    # Gemini debate
    if GEMINI_API_KEY:
        debate_res = debate(signal, price, adjusted, regime.regime.value,
                            h1_trend, h4_trend, h1_rsi, calc_atr(df_m15),
                            mem_stats, anomaly.score, GEMINI_API_KEY)
        log(f"Debate verdict={debate_res['verdict']} "
            f"mult={debate_res['confidence_mult']} "
            f"notes={debate_res.get('notes', '')[:60]}")
    else:
        debate_res = {"verdict": "ABSTAIN", "confidence_mult": 1.0,
                      "notes": "no_api_key", "raw": ""}
        log("[Gemini] Skipped — no API key")

    adjusted *= debate_res["confidence_mult"]
    adjusted = max(0, min(100, adjusted))

    if debate_res["verdict"] == "DISAGREE" and not FORCE_RUN:
        log("AI Council DISAGREE. Skip.")
        _ping("debate_disagree")
        return 0

    # SL/TP
    now_h = datetime.datetime.now(WIB).hour
    sess = 1.35 if (13 <= now_h <= 23 or now_h <= 2) else 1.15
    atr = calc_atr(df_m15)
    sl_base = max(6.0, min(16.0, atr * sess)) + 1.2

    entry = price
    if signal == "BUY":
        sl = entry - sl_base
        tps = [entry + sl_base * m for m in (1.3, 2.2, 3.5, 5.0)]
    else:
        sl = entry + sl_base
        tps = [entry - sl_base * m for m in (1.3, 2.2, 3.5, 5.0)]

    # Save memory + journal
    mem_id = memory.store(df_m15, regime, signal, entry, outcome="OPEN",
                           extra={"grade": grade_info["grade"]})
    trade = {
        "time": datetime.datetime.now(WIB).isoformat(),
        "signal": signal, "price": float(entry),
        "sl": round(sl, 2), "tp1": round(tps[0], 2),
        "tp2": round(tps[1], 2), "tp3": round(tps[2], 2),
        "tp4": round(tps[3], 2),
        "consensus": round(consensus, 2),
        "adjusted_consensus": round(adjusted, 2),
        "grade": grade_info["grade"], "grade_score": grade_info["score"],
        "regime": regime.regime.value,
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
    save_signal(signal, entry)

    # AI insight (local)
    ai = compose_local_insight(signal, adjusted, regime.regime.value,
                                h1_trend, h4_trend, mem_stats, anomaly.score)

    # Caption
    caption = compose_caption(
        signal=signal, entry=entry, sl=sl, tps=tps, atr=atr,
        consensus=adjusted, grade=grade_info["grade"],
        grade_score=grade_info["score"], regime=regime, anomaly=anomaly,
        memory_stats=mem_stats, debate_verdict=debate_res["verdict"],
        debate_notes=debate_res.get("notes", ""), source=source,
        offset=offset, ai_insight=ai,
    )

    # Yahoo delay warning
    if source == "Yahoo GC=F":
        caption += (f"\n⚠️ <i>Yahoo data delay ~10m. "
                    f"Offset={offset:+.2f}. Verify sebelum entry.</i>")

    # Chart + send
    chart = make_chart(df_m15, entry, sl, tps, signal, adjusted, atr)
    if chart and os.path.exists(chart):
        send_photo(TELEGRAM_BOT_TOKEN, caption, chart)
    else:
        send_text(TELEGRAM_BOT_TOKEN, caption, None)

    log(f"✅ SIGNAL SENT: {signal} @ {entry:.2f} ({adjusted:.0f}%) "
        f"grade {grade_info['grade']}")

    # Reflection every 10 trades
    refl = reflect(journal, regime.regime.value, GEMINI_API_KEY, interval=10)
    if refl:
        log("Reflection updated.")

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
