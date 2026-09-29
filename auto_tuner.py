#!/usr/bin/env python3
"""
=============================================================================
AUTO-TUNER - FIND OPTIMAL PARAMETERS
=============================================================================
Cari kombinasi (min_confluence, min_precision, min_grade) yang optimal
via grid search + backtest.

Pakai:
  python auto_tuner.py --deriv --days 60
  python auto_tuner.py --csv data.csv
=============================================================================
"""

import os
import sys
import json
import itertools
import argparse
import datetime
from typing import List, Dict, Tuple
import numpy as np

from backtest import (
    run_backtest, grade_result, print_report,
    load_from_deriv, load_csv, BacktestResult,
)


TUNER_FILE = ".state_cache/tuned_params.json"


# =============================================================================
# FITNESS
# =============================================================================
def fitness(r: BacktestResult, min_trades: int = 15) -> float:
    """
    Skor gabungan untuk ranking parameter.
    Prioritas: profit factor + expectancy + winrate - drawdown.
    """
    if r.total_trades < min_trades:
        return -999.0

    pf = r.profit_factor if r.profit_factor != float("inf") else 5.0
    pf = min(pf, 5.0)

    score = (
        pf * 3.0
        + r.expectancy * 5.0
        + (r.winrate - 40) / 20.0       # bonus kalau WR > 40%
        - r.max_dd_r * 0.1
        + min(r.sharpe, 3.0) * 0.5
    )
    return float(score)


# =============================================================================
# GRID SEARCH
# =============================================================================
def grid_search(df_m15, df_h1, df_h4,
                confluence_range: List[float] = None,
                precision_range: List[float] = None,
                grade_range: List[str] = None,
                verbose: bool = True) -> Tuple[dict, BacktestResult]:
    """
    Grid search parameter. Return (best_params, best_result).
    """
    confluence_range = confluence_range or [60.0, 65.0, 70.0, 75.0]
    precision_range = precision_range or [60.0, 70.0, 80.0]
    grade_range = grade_range or ["A", "A++"]

    results = []
    total_combos = len(confluence_range) * len(precision_range) * len(grade_range)
    idx = 0

    for conf, prec, grade in itertools.product(confluence_range,
                                                 precision_range, grade_range):
        idx += 1
        if verbose:
            print(f"[{idx}/{total_combos}] Testing conf={conf} "
                  f"prec={prec} grade={grade}...", flush=True)
        try:
            r = run_backtest(
                df_m15, df_h1, df_h4,
                min_confluence=conf,
                min_precision=prec,
                min_grade=grade,
                require_mtf=True,
            )
            f = fitness(r)
            results.append({
                "min_confluence": conf,
                "min_precision": prec,
                "min_grade": grade,
                "fitness": round(f, 4),
                "trades": r.total_trades,
                "winrate": r.winrate,
                "pf": r.profit_factor,
                "expectancy": r.expectancy,
                "max_dd": r.max_dd_r,
                "sharpe": r.sharpe,
                "grade": grade_result(r),
                "_result": r,
            })
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    if not results:
        raise RuntimeError("No valid parameter combination")

    results.sort(key=lambda x: -x["fitness"])
    best = results[0]
    if verbose:
        print("\n=== TOP 5 PARAMETER SETS ===")
        for r in results[:5]:
            print(f"  conf={r['min_confluence']} prec={r['min_precision']} "
                  f"grade={r['min_grade']} | fitness={r['fitness']:.2f} "
                  f"WR={r['winrate']}% PF={r['pf']} "
                  f"Exp={r['expectancy']:+.3f} T={r['trades']}")

    best_params = {
        "min_confluence": best["min_confluence"],
        "min_precision": best["min_precision"],
        "min_grade": best["min_grade"],
    }
    return best_params, best["_result"]


def save_tuned(params: dict, result: BacktestResult,
                path: str = TUNER_FILE):
    data = {
        "params": params,
        "backtest_summary": {
            "total_trades": result.total_trades,
            "winrate": result.winrate,
            "profit_factor": result.profit_factor,
            "expectancy": result.expectancy,
            "max_dd_r": result.max_dd_r,
            "sharpe": result.sharpe,
            "grade": grade_result(result),
        },
        "tuned_at": datetime.datetime.utcnow().isoformat(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    print(f"\nSaved: {path}")


def load_tuned(path: str = TUNER_FILE) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--deriv", action="store_true")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default=TUNER_FILE)
    args = ap.parse_args()

    if args.csv:
        df_m15 = load_csv(args.csv)
        df_h1 = df_h4 = df_m15
    elif args.deriv:
        df_m15, df_h1, df_h4 = load_from_deriv(args.days)
    else:
        print("Pakai --csv <path> atau --deriv")
        return 1

    print(f"Data: {len(df_m15)} bars")
    print("\nRunning grid search...")

    best_params, best_result = grid_search(df_m15, df_h1, df_h4)

    print("\n" + "=" * 65)
    print("BEST PARAMETERS")
    print("=" * 65)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("=" * 65)

    print_report(best_result)
    save_tuned(best_params, best_result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
