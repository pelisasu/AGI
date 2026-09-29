#!/usr/bin/env python3
"""
=============================================================================
INSTITUTIONAL ENGINES v31.0
=============================================================================
10 engine profesional + market structure + zone detection.
Semua lokal, numpy + pandas. Tanpa API key.

Engines:
  1. TREND_STACK       - EMA 9/21/50/200 stacked alignment
  2. MARKET_STRUCTURE  - Break of Structure (BOS) / Change of Character (CHoCH)
  3. LIQUIDITY_SWEEP   - Stop hunt di swing high/low
  4. FVG_DETECT        - Fair Value Gap (imbalance)
  5. ORDER_BLOCK       - Order block retest
  6. RSI_DIVERGENCE    - Regular divergence
  7. VOLUME_CLIMAX     - Volume spike + rejection
  8. VOLATILITY_REGIME - ATR compression -> expansion
  9. SESSION_MOMENTUM  - Session open directional drive
  10. MOMENTUM_ROC     - Rate of Change slope
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# INDICATOR HELPERS
# =============================================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr_series(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(p).mean()


def rsi_series(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    ls = -d.where(d < 0, 0).rolling(p).mean()
    rs = g / (ls + 1e-9)
    return 100 - 100 / (1 + rs)


# =============================================================================
# MARKET STRUCTURE
# =============================================================================
@dataclass
class SwingPoint:
    idx: int
    price: float
    kind: str  # "high" or "low"


def find_swings(df: pd.DataFrame, lookback: int = 5,
                limit: int = 20) -> List[SwingPoint]:
    """
    Deteksi swing high/low pakai fractal: titik lebih tinggi/rendah dari
    `lookback` tetangga kiri-kanan.
    """
    swings: List[SwingPoint] = []
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    for i in range(lookback, n - lookback):
        # Swing high
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(i, float(highs[i]), "high"))
        # Swing low
        elif lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(i, float(lows[i]), "low"))

    return swings[-limit:]


def detect_bos_choch(df: pd.DataFrame, swings: List[SwingPoint]) -> Dict:
    """
    Break of Structure (BOS): harga break swing searah trend.
    Change of Character (CHoCH): break swing lawan trend (awal reversal).
    """
    result = {"bos": None, "choch": None, "trend": "NEUTRAL", "level": None}
    if len(swings) < 4:
        return result

    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if not highs or not lows:
        return result

    last_close = float(df["close"].iloc[-1])
    recent_highs = sorted([h.price for h in highs[-3:]], reverse=True)
    recent_lows = sorted([l.price for l in lows[-3:]])

    # Trend dari struktur swing: higher highs + higher lows = uptrend
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price

        if hh and hl:
            result["trend"] = "BULLISH"
        elif lh and ll:
            result["trend"] = "BEARISH"

    # BOS: close break last major swing searah trend
    if result["trend"] == "BULLISH" and recent_highs:
        if last_close > recent_highs[0]:
            result["bos"] = "BULLISH"
            result["level"] = recent_highs[0]
    elif result["trend"] == "BEARISH" and recent_lows:
        if last_close < recent_lows[0]:
            result["bos"] = "BEARISH"
            result["level"] = recent_lows[0]

    # CHoCH: close break lawan trend
    if result["trend"] == "BULLISH" and recent_lows:
        if last_close < recent_lows[-1]:
            result["choch"] = "BEARISH"
    elif result["trend"] == "BEARISH" and recent_highs:
        if last_close > recent_highs[-1]:
            result["choch"] = "BULLISH"

    return result


# =============================================================================
# ZONE DETECTION
# =============================================================================
@dataclass
class Zone:
    kind: str          # "FVG_BULL", "FVG_BEAR", "OB_BULL", "OB_BEAR"
    top: float
    bottom: float
    idx: int
    strength: float    # 0-1

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2


def detect_fvg(df: pd.DataFrame, max_zones: int = 10) -> List[Zone]:
    """
    Fair Value Gap (3-candle imbalance):
    Bullish FVG: high[i-2] < low[i] -> gap antara keduanya
    Bearish FVG: low[i-2] > high[i] -> gap antara keduanya
    """
    zones: List[Zone] = []
    h = df["high"].values
    l = df["low"].values
    n = len(df)

    for i in range(2, n):
        # Bullish FVG
        if h[i - 2] < l[i]:
            gap = l[i] - h[i - 2]
            # Kekuatan: gap besar relatif ke ATR
            atr = np.mean(h[max(0, i - 20):i] - l[max(0, i - 20):i]) + 1e-9
            strength = min(1.0, gap / atr * 2)
            zones.append(Zone("FVG_BULL", top=float(l[i]),
                              bottom=float(h[i - 2]), idx=i, strength=strength))
        # Bearish FVG
        elif l[i - 2] > h[i]:
            gap = l[i - 2] - h[i]
            atr = np.mean(h[max(0, i - 20):i] - l[max(0, i - 20):i]) + 1e-9
            strength = min(1.0, gap / atr * 2)
            zones.append(Zone("FVG_BEAR", top=float(l[i - 2]),
                              bottom=float(h[i]), idx=i, strength=strength))

    return zones[-max_zones:]


def detect_order_blocks(df: pd.DataFrame, max_obs: int = 10) -> List[Zone]:
    """
    Order Block: candle terakhir berlawanan arah sebelum impulsif move.
    Bullish OB: last bearish candle sebelum strong bullish move.
    Bearish OB: last bullish candle sebelum strong bearish move.
    """
    obs: List[Zone] = []
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    if n < 5:
        return obs

    body = np.abs(c - o)
    avg_body = pd.Series(body).rolling(20).mean().values

    for i in range(2, n - 2):
        # Cek impulsif move setelah candle i
        next_body = abs(c[i + 1] - o[i + 1])
        is_impulse = next_body > avg_body[i] * 1.5

        if not is_impulse:
            continue

        # Bullish OB: candle i bearish, candle i+1 bullish besar
        if c[i] < o[i] and c[i + 1] > o[i + 1]:
            top, bottom = float(o[i]), float(c[i])
            if top > bottom:
                obs.append(Zone("OB_BULL", top=top, bottom=bottom,
                                idx=i, strength=0.7))

        # Bearish OB: candle i bullish, candle i+1 bearish besar
        elif c[i] > o[i] and c[i + 1] < o[i + 1]:
            top, bottom = float(c[i]), float(o[i])
            if top > bottom:
                obs.append(Zone("OB_BEAR", top=top, bottom=bottom,
                                idx=i, strength=0.7))

    return obs[-max_obs:]


def find_nearest_zone(zones: List[Zone], price: float,
                      kind_prefix: str, below: bool) -> Optional[Zone]:
    """Cari zone terdekat di bawah / atas harga."""
    candidates = [z for z in zones if z.kind.startswith(kind_prefix)]
    if below:
        candidates = [z for z in candidates if z.top <= price]
        candidates.sort(key=lambda z: price - z.top)
    else:
        candidates = [z for z in candidates if z.bottom >= price]
        candidates.sort(key=lambda z: z.bottom - price)
    return candidates[0] if candidates else None


# =============================================================================
# 10 INSTITUTIONAL ENGINES
# Setiap engine return (signal: -1|0|1, weight: float)
# =============================================================================
def trend_stack(df: pd.DataFrame) -> Tuple[int, float]:
    """EMA 9/21/50/200 stack alignment."""
    try:
        c = df["close"]
        e9 = ema(c, 9).iloc[-1]
        e21 = ema(c, 21).iloc[-1]
        e50 = ema(c, 50).iloc[-1]
        e200 = ema(c, 200).iloc[-1] if len(c) >= 200 else ema(c, min(100, len(c) // 2)).iloc[-1]
        p = c.iloc[-1]

        # Full bull stack
        if p > e9 > e21 > e50 > e200:
            return (1, 2.0)
        # Full bear stack
        if p < e9 < e21 < e50 < e200:
            return (-1, 2.0)
        # Partial
        if p > e21 and e9 > e21:
            return (1, 1.0)
        if p < e21 and e9 < e21:
            return (-1, 1.0)
        return (0, 1.5)
    except Exception:
        return (0, 1.5)


def market_structure(df: pd.DataFrame) -> Tuple[int, float]:
    """BOS/CHoCH direction."""
    try:
        swings = find_swings(df, lookback=4, limit=15)
        s = detect_bos_choch(df, swings)
        if s["bos"] == "BULLISH":
            return (1, 2.0)
        if s["bos"] == "BEARISH":
            return (-1, 2.0)
        if s["choch"] == "BULLISH":
            return (1, 1.5)
        if s["choch"] == "BEARISH":
            return (-1, 1.5)
        if s["trend"] == "BULLISH":
            return (1, 1.0)
        if s["trend"] == "BEARISH":
            return (-1, 1.0)
        return (0, 1.5)
    except Exception:
        return (0, 1.5)


def liquidity_sweep(df: pd.DataFrame) -> Tuple[int, float]:
    """
    Deteksi sweep: harga tembus swing high/low lalu close balik = reversal signal.
    """
    try:
        swings = find_swings(df, lookback=5, limit=20)
        highs = [s.price for s in swings if s.kind == "high"]
        lows = [s.price for s in swings if s.kind == "low"]
        if not highs or not lows:
            return (0, 1.5)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        recent_high = max(highs[-3:])
        recent_low = min(lows[-3:])

        # Bullish sweep: harga tembus low lalu close balik ke atas
        if prev["low"] < recent_low and last["close"] > recent_low:
            return (1, 1.8)
        # Bearish sweep
        if prev["high"] > recent_high and last["close"] < recent_high:
            return (-1, 1.8)
        return (0, 1.8)
    except Exception:
        return (0, 1.5)


def fvg_detect(df: pd.DataFrame) -> Tuple[int, float]:
    """Sinyal dari FVG terdekat ke harga."""
    try:
        zones = detect_fvg(df, max_zones=5)
        if not zones:
            return (0, 1.3)
        price = float(df["close"].iloc[-1])
        last = zones[-1]
        # Kalau harga di dalam FVG bullish
        if last.kind == "FVG_BULL" and last.bottom <= price <= last.top:
            return (1, 1.5)
        if last.kind == "FVG_BEAR" and last.bottom <= price <= last.top:
            return (-1, 1.5)
        # Kalau FVG baru terbentuk
        if last.idx >= len(df) - 5:
            return (1 if last.kind == "FVG_BULL" else -1, 1.2)
        return (0, 1.3)
    except Exception:
        return (0, 1.3)


def order_block(df: pd.DataFrame) -> Tuple[int, float]:
    """Retest order block terdekat."""
    try:
        obs = detect_order_blocks(df, max_obs=5)
        if not obs:
            return (0, 1.5)
        price = float(df["close"].iloc[-1])
        last = obs[-1]
        # Cek apakah harga di zona OB
        in_zone = last.bottom <= price <= last.top
        if not in_zone:
            return (0, 1.3)
        # Konfirmasi candle rejection
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        rng = c["high"] - c["low"] + 1e-9
        is_rejection = body < rng * 0.5

        if last.kind == "OB_BULL" and is_rejection:
            return (1, 1.6)
        if last.kind == "OB_BEAR" and is_rejection:
            return (-1, 1.6)
        return (0, 1.5)
    except Exception:
        return (0, 1.5)


def rsi_divergence(df: pd.DataFrame) -> Tuple[int, float]:
    """RSI divergence pada 20 bar terakhir."""
    try:
        if len(df) < 40:
            return (0, 1.4)
        r = rsi_series(df["close"], 14)
        swings = find_swings(df, lookback=3, limit=10)
        highs = [s for s in swings if s.kind == "high"][-2:]
        lows = [s for s in swings if s.kind == "low"][-2:]

        # Bearish divergence: HH price, LH RSI
        if len(highs) == 2:
            price_hh = highs[-1].price > highs[-2].price
            try:
                rsi_now = float(r.iloc[highs[-1].idx])
                rsi_prev = float(r.iloc[highs[-2].idx])
                rsi_lh = rsi_now < rsi_prev
                if price_hh and rsi_lh and rsi_now > 60:
                    return (-1, 1.7)
            except Exception:
                pass

        # Bullish divergence: LL price, HL RSI
        if len(lows) == 2:
            price_ll = lows[-1].price < lows[-2].price
            try:
                rsi_now = float(r.iloc[lows[-1].idx])
                rsi_prev = float(r.iloc[lows[-2].idx])
                rsi_hl = rsi_now > rsi_prev
                if price_ll and rsi_hl and rsi_now < 40:
                    return (1, 1.7)
            except Exception:
                pass
        return (0, 1.4)
    except Exception:
        return (0, 1.4)


def volume_climax(df: pd.DataFrame) -> Tuple[int, float]:
    """Volume spike + rejection wick."""
    try:
        v = df["volume"]
        if v.iloc[-1] < v.rolling(20).mean().iloc[-1] * 2.0:
            return (0, 1.2)
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        rng = c["high"] - c["low"] + 1e-9
        upper_wick = c["high"] - max(c["close"], c["open"])
        lower_wick = min(c["close"], c["open"]) - c["low"]

        # Bullish climax: volume spike + lower wick besar
        if lower_wick > body * 2 and lower_wick > upper_wick:
            return (1, 1.4)
        # Bearish climax
        if upper_wick > body * 2 and upper_wick > lower_wick:
            return (-1, 1.4)
        return (0, 1.2)
    except Exception:
        return (0, 1.2)


def volatility_regime(df: pd.DataFrame) -> Tuple[int, float]:
    """ATR compression -> expansion + direction."""
    try:
        a = atr_series(df, 14)
        if len(a) < 30 or pd.isna(a.iloc[-1]):
            return (0, 1.2)
        atr_now = a.iloc[-1]
        atr_avg = a.rolling(30).mean().iloc[-1]
        atr_prev = a.iloc[-5]

        expanding = atr_now > atr_prev * 1.15 and atr_now > atr_avg
        if not expanding:
            return (0, 1.2)

        # Arah dari body candle terakhir
        c = df.iloc[-1]
        if c["close"] > c["open"]:
            return (1, 1.3)
        if c["close"] < c["open"]:
            return (-1, 1.3)
        return (0, 1.2)
    except Exception:
        return (0, 1.2)


def session_momentum(df: pd.DataFrame) -> Tuple[int, float]:
    """Momentum sesi: 4 bar pertama saat London/NY open."""
    try:
        import datetime as _dt
        if not isinstance(df.index, pd.DatetimeIndex):
            return (0, 1.0)
        last = df.index[-1]
        h = last.hour
        # London open 07-09 UTC, NY open 12-14 UTC
        in_session_open = (7 <= h <= 9) or (12 <= h <= 14)
        if not in_session_open or len(df) < 5:
            return (0, 1.0)
        # Bandingkan 4 bar terakhir vs 4 bar sebelumnya
        recent = df["close"].iloc[-4:].mean()
        prev = df["close"].iloc[-8:-4].mean()
        if recent > prev * 1.0005:
            return (1, 1.4)
        if recent < prev * 0.9995:
            return (-1, 1.4)
        return (0, 1.0)
    except Exception:
        return (0, 1.0)


def momentum_roc(df: pd.DataFrame) -> Tuple[int, float]:
    """Rate of Change slope."""
    try:
        c = df["close"]
        if len(c) < 20:
            return (0, 1.2)
        roc = (c.iloc[-1] - c.iloc[-10]) / (c.iloc[-10] + 1e-9)
        roc_prev = (c.iloc[-5] - c.iloc[-15]) / (c.iloc[-15] + 1e-9)
        # Akselerasi
        if roc > 0.001 and roc > roc_prev:
            return (1, 1.3)
        if roc < -0.001 and roc < roc_prev:
            return (-1, 1.3)
        return (0, 1.2)
    except Exception:
        return (0, 1.2)


# =============================================================================
# REGISTRY
# =============================================================================
ENGINES = [
    ("TREND_STACK",       trend_stack),
    ("MARKET_STRUCTURE",  market_structure),
    ("LIQUIDITY_SWEEP",   liquidity_sweep),
    ("FVG_DETECT",        fvg_detect),
    ("ORDER_BLOCK",       order_block),
    ("RSI_DIVERGENCE",    rsi_divergence),
    ("VOLUME_CLIMAX",     volume_climax),
    ("VOLATILITY_REGIME", volatility_regime),
    ("SESSION_MOMENTUM",  session_momentum),
    ("MOMENTUM_ROC",      momentum_roc),
]
