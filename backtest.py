#!/usr/bin/env python3
"""
=============================================================================
BACKTEST HARNESS v31.3 (LEVEL 3)
=============================================================================
Fix: Yahoo fetch lokal tanpa truncate 300 bar.
=============================================================================
"""

import os
import sys
import json
import argparse
import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from engines import ENGINES, find_swings, detect_fvg, detect_order_blocks
from precision_entry import calculate_precise_entry
from agi_core import detect_regime, grade_signal, grade_at_least


@dataclass
class BacktestTrade:
    entry_time: str
    exit_time: str
    signal: str
    entry: float
    sl: float
    tp1: float
    exit_price: float
    exit_reason: str
    pnl_r: float
    regime: str
    grade: str
    precision_score: float
    consensus: float
    engine_states: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    winrate: float = 0.0
    total_r: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    max_dd_r: float = 0.0
    sharpe: float = 0.0
    expectancy: float = 0.0
    avg_precision: float = 0.0
    regime_breakdown: dict = field(default_factory=dict)
    engine_accuracy: dict = field(default_factory=dict)
    trades: List[dict] = field(default_factory=list)


# =============================================================================
# YAHOO DIRECT FETCH (TANPA TRUNCATE)
# =============================================================================
def _fetch_yahoo(symbol: str = "GC=F", period: str = "60d",
                  interval: str = "15m") -> Optional[pd.DataFrame]:
    """Fetch Yahoo tanpa truncate. Return DataFrame dengan DatetimeIndex UTC."""
    try:
        raw = yf.Ticker(symbol).history(period=period, interval=interval,
                                          auto_adjust=False)
        if raw is None or len(raw) < 50:
            return None
        raw = raw.reset_index()
        dc = next((c for c in raw.columns if "date" in c.lower()),
                  raw.columns[0])
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
        raw = raw.set_index("datetime").sort_index()
        return raw[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"  Yahoo error: {e}")
        return None


# =============================================================================
# ENGINE SCORING / EXIT SIM / FINALIZE / REPORT (sama seperti v31.2)
# =============================================================================
def _atr(df, p=14):
    try:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                       axis=1).max(axis=1)
        v = tr.rolling(p).mean().iloc[-1]
        return float(v) if not pd.isna(v) and v > 0 else 8.0
    except Exception:
        return 8.0


def _trend_label(df):
    try:
        if df is None or len(df) < 50:
            return "NEUTRAL"
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


def _score_engines(df):
    buy_w = sell_w = 0.0
    states = {}
    for name, fn in ENGINES:
        try:
            sc, w = fn(df)
            states[name] = {"sc": sc, "weight_used": w}
            if sc > 0:
                buy_w += w * abs(sc)
            elif sc < 0:
                sell_w += w * abs(sc)
        except Exception:
            states[name] = {"sc": 0, "weight_used": 1.0}
    return buy_w, sell_w, states


def _simulate_exit(df, entry_idx, signal, entry, sl, tp1, max_bars=24):
    for i in range(entry_idx + 1, min(entry_idx + 1 + max_bars, len(df))):
        row = df.iloc[i]
        if signal == "BUY":
            if row["low"] <= sl:
                return (sl, "SL", df.index[i].isoformat())
            if row["high"] >= tp1:
                return (tp1, "TP1", df.index[i].isoformat())
        else:
            if row["high"] >= sl:
                return (sl, "SL", df.index[i].isoformat())
            if row["low"] <= tp1:
                return (tp1, "TP1", df.index[i].isoformat())
    last_idx = min(entry_idx + max_bars, len(df) - 1)
    return (float(df.iloc[last_idx]["close"]), "TIMEOUT",
            df.index[last_idx].isoformat())


def run_backtest(df_m15, df_h1=None, df_h4=None,
                  min_confluence=65.0, min_precision=70.0,
                  min_grade="A", warmup=250, cooldown_bars=4,
                  max_bars=24, require_mtf=True):
    result = BacktestResult()
    if df_m15 is None or len(df_m15) < warmup + 20:
        return result
    if df_h1 is None:
        df_h1 = df_m15
    if df_h4 is None:
        df_h4 = df_h1

    last_entry_idx = -10_000
    i = warmup

    while i < len(df_m15) - 1:
        if i - last_entry_idx < cooldown_bars:
            i += 1
            continue

        window = df_m15.iloc[max(0, i - 250):i + 1]
        if len(window) < 100:
            i += 1
            continue

        try:
            regime_state = detect_regime(window)
        except Exception:
            i += 1
            continue

        h1_slice = df_h1[df_h1.index <= df_m15.index[i]] \
            if isinstance(df_h1.index, pd.DatetimeIndex) else df_h1
        h4_slice = df_h4[df_h4.index <= df_m15.index[i]] \
            if isinstance(df_h4.index, pd.DatetimeIndex) else df_h4
        h1_trend = _trend_label(h1_slice)
        h4_trend = _trend_label(h4_slice)

        buy_w, sell_w, states = _score_engines(window)
        if "BULLISH" in h1_trend:
            buy_w += 2.5
        elif "BEARISH" in h1_trend:
            sell_w += 2.5
        if "BULLISH" in h4_trend:
            buy_w += 1.0
        elif "BEARISH" in h4_trend:
            sell_w += 1.0

        total = buy_w + sell_w
        if total == 0 or buy_w == sell_w:
            i += 1
            continue

        consensus = max(buy_w, sell_w) / total * 100
        if consensus < min_confluence:
            i += 1
            continue

        signal = "BUY" if buy_w > sell_w else "SELL"

        if require_mtf:
            aligned = ((signal == "BUY" and "BULLISH" in h1_trend) or
                       (signal == "SELL" and "BEARISH" in h1_trend))
            if not aligned:
                i += 1
                continue

        atr = _atr(window)
        try:
            grade_info = grade_signal(consensus, signal, states,
                                       h1_trend, h4_trend, atr,
                                       {"n": 0, "winrate": 50}, 0.0)
        except Exception:
            i += 1
            continue

        if not grade_at_least(grade_info["grade"], min_grade):
            i += 1
            continue

        try:
            entry_data = calculate_precise_entry(
                window, signal, float(window["close"].iloc[-1]),
                h4_trend, h1_trend, consensus, atr
            )
        except Exception:
            i += 1
            continue

        if entry_data.precision_score < min_precision:
            i += 1
            continue

        exit_price, reason, exit_time = _simulate_exit(
            df_m15, i, signal, entry_data.entry_ideal,
            entry_data.sl, entry_data.tp1, max_bars=max_bars
        )

        risk = abs(entry_data.entry_ideal - entry_data.sl)
        if risk <= 0:
            i += 1
            continue
        if signal == "BUY":
            pnl_r = (exit_price - entry_data.entry_ideal) / risk
        else:
            pnl_r = (entry_data.entry_ideal - exit_price) / risk

        trade = BacktestTrade(
            entry_time=df_m15.index[i].isoformat(),
            exit_time=exit_time,
            signal=signal,
            entry=round(entry_data.entry_ideal, 2),
            sl=round(entry_data.sl, 2),
            tp1=round(entry_data.tp1, 2),
            exit_price=round(exit_price, 2),
            exit_reason=reason,
            pnl_r=round(pnl_r, 4),
            regime=regime_state.regime.value,
            grade=grade_info["grade"],
            precision_score=entry_data.precision_score,
            consensus=round(consensus, 2),
            engine_states={k: v["sc"] for k, v in states.items()},
        )
        result.trades.append(asdict(trade))
        last_entry_idx = i
        i += 1

    _finalize(result)
    return result


def _finalize(r):
    if not r.trades:
        return
    rs = np.array([t["pnl_r"] for t in r.trades])
    r.total_trades = len(rs)
    r.wins = int((rs > 0).sum())
    r.losses = int((rs < 0).sum())
    r.winrate = round(r.wins / r.total_trades * 100, 2)
    r.total_r = round(float(rs.sum()), 3)
    r.avg_r = round(float(rs.mean()), 4)
    r.expectancy = r.avg_r
    r.avg_precision = round(float(np.mean([t["precision_score"]
                                             for t in r.trades])), 2)

    gw = float(rs[rs > 0].sum()) if (rs > 0).any() else 0.0
    gl = float(abs(rs[rs < 0].sum())) if (rs < 0).any() else 0.0
    r.profit_factor = round(gw / gl, 3) if gl > 0 else (
        float("inf") if gw > 0 else 0.0)

    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    r.max_dd_r = round(float(dd.max()) if len(dd) else 0.0, 3)

    if rs.std() > 1e-9:
        r.sharpe = round(float(rs.mean() / rs.std() * np.sqrt(252)), 3)

    rb = {}
    for t in r.trades:
        reg = t["regime"]
        d = rb.setdefault(reg, {"n": 0, "wins": 0, "r_sum": 0.0})
        d["n"] += 1
        if t["pnl_r"] > 0:
            d["wins"] += 1
        d["r_sum"] += t["pnl_r"]
    for reg, d in rb.items():
        d["winrate"] = round(d["wins"] / d["n"] * 100, 1)
        d["r_sum"] = round(d["r_sum"], 2)
    r.regime_breakdown = rb

    ea = {}
    for t in r.trades:
        signal = t["signal"]
        result = "WIN" if t["pnl_r"] > 0 else "LOSS"
        for name, sc in t["engine_states"].items():
            if sc == 0:
                continue
            d = ea.setdefault(name, {"n": 0, "correct": 0})
            d["n"] += 1
            was_bull = sc > 0
            should_bull = (signal == "BUY")
            if (was_bull == should_bull and result == "WIN") or \
               (was_bull != should_bull and result == "LOSS"):
                d["correct"] += 1
    for name, d in ea.items():
        d["accuracy"] = round(d["correct"] / d["n"] * 100, 1)
    r.engine_accuracy = ea


def grade_result(r):
    if r.total_trades < 20:
        return "INSUFFICIENT"
    score = 0
    if r.winrate >= 50: score += 2
    elif r.winrate >= 42: score += 1
    if r.profit_factor >= 1.5: score += 3
    elif r.profit_factor >= 1.2: score += 2
    elif r.profit_factor >= 1.0: score += 1
    if r.expectancy > 0.15: score += 2
    elif r.expectancy > 0: score += 1
    if r.sharpe >= 1.5: score += 2
    elif r.sharpe >= 0.8: score += 1
    if r.max_dd_r < 10: score += 2
    elif r.max_dd_r < 20: score += 1

    if score >= 10: return "A+++"
    if score >= 8: return "A++"
    if score >= 6: return "A"
    if score >= 4: return "B"
    if score >= 2: return "C"
    return "D"


def print_report(r):
    print("\n" + "=" * 65)
    print("BACKTEST REPORT")
    print("=" * 65)
    print(f"Trades        : {r.total_trades}")
    print(f"Winrate       : {r.winrate}% (W{r.wins}/L{r.losses})")
    print(f"Total R       : {r.total_r:+.2f}")
    print(f"Avg R         : {r.avg_r:+.4f}")
    print(f"Expectancy    : {r.expectancy:+.4f}R")
    print(f"Profit Factor : {r.profit_factor}")
    print(f"Max DD        : {r.max_dd_r:.2f}R")
    print(f"Sharpe        : {r.sharpe}")
    print(f"Avg Precision : {r.avg_precision}")
    print(f"\nGRADE: {grade_result(r)}")
    print("=" * 65)

    if r.regime_breakdown:
        print("\nPer Regime:")
        for reg, d in sorted(r.regime_breakdown.items(),
                             key=lambda x: -x[1]["r_sum"]):
            print(f"  {reg:16s} n={d['n']:3d} wr={d['winrate']:5.1f}% "
                  f"R={d['r_sum']:+.2f}")

    if r.engine_accuracy:
        print("\nPer Engine Accuracy:")
        for name, d in sorted(r.engine_accuracy.items(),
                              key=lambda x: -x[1]["accuracy"]):
            print(f"  {name:22s} acc={d['accuracy']:5.1f}% n={d['n']}")


def save_report(r, path):
    data = asdict(r)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nSaved: {path}")


# =============================================================================
# DATA LOADER (pakai Yahoo langsung, tanpa truncate)
# =============================================================================
def load_from_deriv(days=60):
    """Fetch data Yahoo GC=F tanpa truncate 300 bar."""
    print(f"Loading {days} days data...")

    # --- M15: Yahoo max 60d untuk 15m ---
    yf_days = min(days, 55)
    print(f"  Fetching M15 ({yf_days}d)...")
    df_m15 = _fetch_yahoo("GC=F", f"{yf_days}d", "15m")
    if df_m15 is None or len(df_m15) < 500:
        raise RuntimeError(f"M15 fetch gagal, hanya dapat "
                            f"{len(df_m15) if df_m15 is not None else 0} bars")
    print(f"  M15 OK: {len(df_m15)} bars")

    # --- H1: Yahoo max 730d ---
    print(f"  Fetching H1...")
    df_h1 = _fetch_yahoo("GC=F", "180d", "1h")
    if df_h1 is None or len(df_h1) < 100:
        print("  H1 gagal, pakai M15")
        df_h1 = df_m15
    else:
        print(f"  H1 OK: {len(df_h1)} bars")

    # --- H4: pakai 1d sebagai proxy (Yahoo gak ada 4h) ---
    print(f"  Fetching H4 (via 1d)...")
    df_h4 = _fetch_yahoo("GC=F", "2y", "1d")
    if df_h4 is None or len(df_h4) < 50:
        print("  H4 gagal, pakai H1")
        df_h4 = df_h1
    else:
        print(f"  H4 OK: {len(df_h4)} bars")

    return df_m15, df_h1, df_h4


def load_csv(path):
    df = pd.read_csv(path)
    dc = next((c for c in df.columns
               if "date" in c.lower() or "time" in c.lower()), df.columns[0])
    df["datetime"] = pd.to_datetime(df[dc], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    if "volume" not in df.columns:
        df["volume"] = 100.0
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--deriv", action="store_true")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--min-score", type=float, default=65.0)
    ap.add_argument("--min-precision", type=float, default=70.0)
    ap.add_argument("--min-grade", default="A")
    ap.add_argument("--out", default="backtest_report.json")
    ap.add_argument("--no-mtf", action="store_true")
    args = ap.parse_args()

    if args.csv:
        df_m15 = load_csv(args.csv)
        df_h1 = df_m15
        df_h4 = df_m15
    elif args.deriv:
        df_m15, df_h1, df_h4 = load_from_deriv(args.days)
    else:
        print("Pakai --csv <path> atau --deriv")
        return 1

    print(f"Data: {len(df_m15)} bars | "
          f"{df_m15.index[0]} -> {df_m15.index[-1]}")

    r = run_backtest(df_m15, df_h1, df_h4,
                      min_confluence=args.min_score,
                      min_precision=args.min_precision,
                      min_grade=args.min_grade,
                      require_mtf=not args.no_mtf)
    print_report(r)
    save_report(r, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
