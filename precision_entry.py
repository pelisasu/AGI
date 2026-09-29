#!/usr/bin/env python3
"""
=============================================================================
PRECISION ENTRY SYSTEM v31.0
=============================================================================
Hitung Entry / SL / TP presisi berbasis:
  - FVG zone (limit order di zona)
  - Order Block
  - Swing structure (structural SL)
  - Liquidity levels (structural TP)
  - Round numbers (psychological)
  - ATR untuk buffer saja

Output: Presisi OP dengan confidence score 0-100
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from engines import (
    find_swings, detect_fvg, detect_order_blocks,
    find_nearest_zone, atr_series, Zone,
)


@dataclass
class PrecisionEntry:
    # Zona entry presisi
    entry_low: float          # batas bawah limit order
    entry_high: float         # batas atas limit order
    entry_ideal: float        # harga limit ideal (mid FVG / OB 50%)
    entry_type: str           # "FVG_FILL" | "OB_RETEST" | "STRUCTURE_PULLBACK"

    # SL struktural
    sl: float
    sl_reason: str

    # TP struktural berjenjang
    tp1: float
    tp2: float
    tp3: float
    tp4: float
    tp_reasons: List[str]

    # Risk metrics
    risk_points: float
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float

    # Confidence
    precision_score: float    # 0-100
    precision_grade: str      # A+++ | A++ | A | B | C | D
    notes: List[str] = field(default_factory=list)


def _round_number_levels(price: float, count: int = 3,
                          step: float = 10.0) -> List[float]:
    """Level angka bulat psikologis (50, 100, dll)."""
    base = round(price / step) * step
    return [base + i * step for i in range(-count, count + 1)]


def _find_swing_above(df: pd.DataFrame, price: float,
                       swings) -> Optional[float]:
    highs = sorted([s.price for s in swings
                    if s.kind == "high" and s.price > price])
    return highs[0] if highs else None


def _find_swing_below(df: pd.DataFrame, price: float,
                       swings) -> Optional[float]:
    lows = sorted([s.price for s in swings
                   if s.kind == "low" and s.price < price], reverse=True)
    return lows[0] if lows else None


def _grade_precision(score: float) -> str:
    if score >= 90:
        return "A+++"
    if score >= 80:
        return "A++"
    if score >= 70:
        return "A"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def calculate_precise_entry(df: pd.DataFrame, signal: str,
                             base_price: float,
                             h4_trend: str, h1_trend: str,
                             consensus: float,
                             atr_value: float) -> PrecisionEntry:
    """
    Hitung entry presisi. Signal: "BUY" atau "SELL".
    """
    notes: List[str] = []
    score = 0.0

    swings = find_swings(df, lookback=4, limit=25)
    fvgs = detect_fvg(df, max_zones=10)
    obs = detect_order_blocks(df, max_obs=10)

    # -------------------------------------------------------------------------
    # 1. TENTUKAN ZONA ENTRY
    # -------------------------------------------------------------------------
    entry_type = "STRUCTURE_PULLBACK"
    entry_ideal = base_price
    entry_low = base_price
    entry_high = base_price

    if signal == "BUY":
        # Cari FVG bullish di bawah harga (support zone)
        fvg = find_nearest_zone(fvgs, base_price, "FVG_BULL", below=True)
        ob = find_nearest_zone(obs, base_price, "OB_BULL", below=True)

        candidates = []
        if fvg:
            candidates.append(("FVG_FILL", fvg, fvg.strength + 0.2))
        if ob:
            candidates.append(("OB_RETEST", ob, ob.strength + 0.15))

        if candidates:
            # Pilih yang terkuat, tapi zona tidak lebih dari 2*ATR dari harga
            candidates = [(t, z, s) for t, z, s in candidates
                          if (base_price - z.top) <= atr_value * 2.0]
            if candidates:
                candidates.sort(key=lambda x: -x[2])
                entry_type, zone, _ = candidates[0]
                entry_high = min(zone.top, base_price)
                entry_low = zone.bottom
                entry_ideal = (entry_high + entry_low) / 2.0
                notes.append(f"Entry di {entry_type}: {entry_low:.2f}-{entry_high:.2f}")
                score += 15 if entry_type == "FVG_FILL" else 12
        else:
            # Fallback: pullback ke EMA 21
            e21 = df["close"].ewm(span=21).mean().iloc[-1]
            if base_price > e21:
                entry_ideal = float(e21)
                entry_low = entry_ideal - atr_value * 0.2
                entry_high = entry_ideal + atr_value * 0.2
                entry_type = "STRUCTURE_PULLBACK"
                notes.append(f"Pullback ke EMA21: {entry_ideal:.2f}")
                score += 5

    else:  # SELL
        fvg = find_nearest_zone(fvgs, base_price, "FVG_BEAR", below=False)
        ob = find_nearest_zone(obs, base_price, "OB_BEAR", below=False)

        candidates = []
        if fvg:
            candidates.append(("FVG_FILL", fvg, fvg.strength + 0.2))
        if ob:
            candidates.append(("OB_RETEST", ob, ob.strength + 0.15))

        if candidates:
            candidates = [(t, z, s) for t, z, s in candidates
                          if (z.bottom - base_price) <= atr_value * 2.0]
            if candidates:
                candidates.sort(key=lambda x: -x[2])
                entry_type, zone, _ = candidates[0]
                entry_low = max(zone.bottom, base_price)
                entry_high = zone.top
                entry_ideal = (entry_high + entry_low) / 2.0
                notes.append(f"Entry di {entry_type}: {entry_low:.2f}-{entry_high:.2f}")
                score += 15 if entry_type == "FVG_FILL" else 12
        else:
            e21 = df["close"].ewm(span=21).mean().iloc[-1]
            if base_price < e21:
                entry_ideal = float(e21)
                entry_low = entry_ideal - atr_value * 0.2
                entry_high = entry_ideal + atr_value * 0.2
                entry_type = "STRUCTURE_PULLBACK"
                notes.append(f"Pullback ke EMA21: {entry_ideal:.2f}")
                score += 5

    # -------------------------------------------------------------------------
    # 2. STOP LOSS STRUKTURAL
    # -------------------------------------------------------------------------
    buffer = atr_value * 0.25

    if signal == "BUY":
        # SL di bawah swing low terdekat
        swing_low = _find_swing_below(df, entry_low, swings)
        if swing_low:
            sl = swing_low - buffer
            sl_reason = f"Below swing low {swing_low:.2f}"
            score += 10
        else:
            sl = entry_low - atr_value * 1.2
            sl_reason = f"ATR-based {atr_value*1.2:.2f}"
            score += 3

        # Pastikan risk minimal
        if entry_ideal - sl < atr_value * 0.5:
            sl = entry_ideal - atr_value * 0.8
            sl_reason += " (widened to min risk)"
    else:  # SELL
        swing_high = _find_swing_above(df, entry_high, swings)
        if swing_high:
            sl = swing_high + buffer
            sl_reason = f"Above swing high {swing_high:.2f}"
            score += 10
        else:
            sl = entry_high + atr_value * 1.2
            sl_reason = f"ATR-based {atr_value*1.2:.2f}"
            score += 3

        if sl - entry_ideal < atr_value * 0.5:
            sl = entry_ideal + atr_value * 0.8
            sl_reason += " (widened to min risk)"

    risk_points = abs(entry_ideal - sl)

    # -------------------------------------------------------------------------
    # 3. TAKE PROFIT STRUKTURAL
    # -------------------------------------------------------------------------
    tp_reasons: List[str] = []

    if signal == "BUY":
        # TP1: swing high terdekat
        swing_high_1 = _find_swing_above(df, entry_high, swings)
        if swing_high_1:
            tp1 = swing_high_1
            tp_reasons.append(f"TP1: swing high {tp1:.2f}")
        else:
            tp1 = entry_ideal + risk_points * 1.5
            tp_reasons.append(f"TP1: 1.5R")

        # TP2: FVG bearish di atas
        fvg_above = find_nearest_zone(fvgs, entry_high, "FVG_BEAR", below=False)
        if fvg_above:
            tp2 = fvg_above.bottom
            tp_reasons.append(f"TP2: FVG top {tp2:.2f}")
        else:
            tp2 = entry_ideal + risk_points * 2.5
            tp_reasons.append(f"TP2: 2.5R")

        # TP3: round number / swing major
        round_levels = _round_number_levels(entry_ideal, 3, 10.0)
        round_above = sorted([r for r in round_levels if r > tp2])
        if round_above:
            tp3 = round_above[0]
            tp_reasons.append(f"TP3: round number {tp3:.2f}")
        else:
            tp3 = entry_ideal + risk_points * 4.0
            tp_reasons.append("TP3: 4.0R")

        # TP4: extension
        tp4 = entry_ideal + risk_points * 6.0
        tp_reasons.append("TP4: 6.0R extension")

    else:  # SELL
        swing_low_1 = _find_swing_below(df, entry_low, swings)
        if swing_low_1:
            tp1 = swing_low_1
            tp_reasons.append(f"TP1: swing low {tp1:.2f}")
        else:
            tp1 = entry_ideal - risk_points * 1.5
            tp_reasons.append(f"TP1: 1.5R")

        fvg_below = find_nearest_zone(fvgs, entry_low, "FVG_BULL", below=True)
        if fvg_below:
            tp2 = fvg_below.top
            tp_reasons.append(f"TP2: FVG bottom {tp2:.2f}")
        else:
            tp2 = entry_ideal - risk_points * 2.5
            tp_reasons.append(f"TP2: 2.5R")

        round_levels = _round_number_levels(entry_ideal, 3, 10.0)
        round_below = sorted([r for r in round_levels if r < tp2], reverse=True)
        if round_below:
            tp3 = round_below[0]
            tp_reasons.append(f"TP3: round number {tp3:.2f}")
        else:
            tp3 = entry_ideal - risk_points * 4.0
            tp_reasons.append("TP3: 4.0R")

        tp4 = entry_ideal - risk_points * 6.0
        tp_reasons.append("TP4: 6.0R extension")

    # -------------------------------------------------------------------------
    # 4. RR CALC
    # -------------------------------------------------------------------------
    risk = max(risk_points, 0.01)
    rr1 = abs(tp1 - entry_ideal) / risk
    rr2 = abs(tp2 - entry_ideal) / risk
    rr3 = abs(tp3 - entry_ideal) / risk

    # -------------------------------------------------------------------------
    # 5. PRECISION SCORE
    # -------------------------------------------------------------------------
    # Multi-TF alignment
    if signal == "BUY":
        if "BULLISH" in h4_trend and "BULLISH" in h1_trend:
            score += 25
            notes.append("MTF aligned bullish")
        elif "BULLISH" in h1_trend:
            score += 12
    else:
        if "BEARISH" in h4_trend and "BEARISH" in h1_trend:
            score += 25
            notes.append("MTF aligned bearish")
        elif "BEARISH" in h1_trend:
            score += 12

    # Consensus strength
    score += min(15, max(0, (consensus - 60) / 40 * 15))

    # RR check
    if rr1 >= 1.5:
        score += 10
    if rr2 >= 2.5:
        score += 10
    if rr3 >= 3.5:
        score += 5

    # Zone strength
    if entry_type == "FVG_FILL":
        score += 5
    elif entry_type == "OB_RETEST":
        score += 3

    score = max(0, min(100, score))
    grade = _grade_precision(score)

    return PrecisionEntry(
        entry_low=round(entry_low, 2),
        entry_high=round(entry_high, 2),
        entry_ideal=round(entry_ideal, 2),
        entry_type=entry_type,
        sl=round(sl, 2),
        sl_reason=sl_reason,
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        tp3=round(tp3, 2),
        tp4=round(tp4, 2),
        tp_reasons=tp_reasons,
        risk_points=round(risk_points, 2),
        rr_tp1=round(rr1, 2),
        rr_tp2=round(rr2, 2),
        rr_tp3=round(rr3, 2),
        precision_score=round(score, 2),
        precision_grade=grade,
        notes=notes,
    )
