#!/usr/bin/env python3
"""
=============================================================================
ENGINE TRACKER - PER (ENGINE x REGIME) ACCURACY
=============================================================================
Track performa tiap engine per regime. Kalau accuracy < threshold setelah
N sample, auto-disable engine di regime tersebut.

Auto-disable = engine tetap dihitung tapi bobotnya di-set 0.
=============================================================================
"""

import os
import json
import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict


TRACKER_FILE = ".state_cache/engine_tracker.json"


@dataclass
class EngineStat:
    n: int = 0
    correct: int = 0
    recent_correct: list = field(default_factory=list)  # sliding window
    disabled: bool = False
    disabled_at: Optional[str] = None
    disabled_reason: str = ""

    def accuracy(self) -> float:
        if self.n == 0:
            return 0.5
        return self.correct / self.n

    def recent_accuracy(self, window: int = 30) -> float:
        if len(self.recent_correct) < 5:
            return self.accuracy()
        window_data = self.recent_correct[-window:]
        return sum(window_data) / len(window_data)


class EngineTracker:
    """
    Menyimpan statistik tiap engine per regime.
    Struktur: {regime: {engine_name: EngineStat}}
    """

    def __init__(self, path: str = TRACKER_FILE,
                 min_samples: int = 15,
                 disable_threshold: float = 0.38,
                 reenable_threshold: float = 0.50):
        self.path = path
        self.min_samples = min_samples
        self.disable_threshold = disable_threshold
        self.reenable_threshold = reenable_threshold
        self.data: Dict[str, Dict[str, EngineStat]] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                raw = json.load(f)
            for regime, engines in raw.items():
                self.data[regime] = {}
                for name, stat_dict in engines.items():
                    self.data[regime][name] = EngineStat(**stat_dict)
        except Exception:
            self.data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        out = {}
        for regime, engines in self.data.items():
            out[regime] = {name: asdict(s) for name, s in engines.items()}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, self.path)

    def update(self, regime: str, engine_name: str,
               engine_signal: int, trade_signal: str, trade_result: str):
        """
        Update statistik engine. engine_signal: -1/0/1.
        trade_result: WIN/LOSS.
        """
        if engine_signal == 0:
            return
        if trade_result not in ("WIN", "LOSS"):
            return

        regime_stats = self.data.setdefault(regime, {})
        stat = regime_stats.setdefault(engine_name, EngineStat())

        was_bull = engine_signal > 0
        should_bull = (trade_signal == "BUY")
        engine_correct = (was_bull == should_bull)
        # Engine benar kalau prediksinya konsisten dengan outcome
        success = (engine_correct and trade_result == "WIN") or \
                  (not engine_correct and trade_result == "LOSS")

        stat.n += 1
        if success:
            stat.correct += 1
        stat.recent_correct.append(1 if success else 0)
        if len(stat.recent_correct) > 100:
            stat.recent_correct = stat.recent_correct[-100:]

        # Auto-disable logic
        if stat.n >= self.min_samples:
            recent_acc = stat.recent_accuracy(window=30)
            if recent_acc < self.disable_threshold and not stat.disabled:
                stat.disabled = True
                stat.disabled_at = datetime.datetime.utcnow().isoformat()
                stat.disabled_reason = (
                    f"acc={recent_acc:.2%} < {self.disable_threshold:.2%} "
                    f"(n={stat.n})"
                )
            elif recent_acc >= self.reenable_threshold and stat.disabled:
                stat.disabled = False
                stat.disabled_at = None
                stat.disabled_reason = "reenabled"

    def is_disabled(self, regime: str, engine_name: str) -> bool:
        stat = self.data.get(regime, {}).get(engine_name)
        return stat.disabled if stat else False

    def get_weight_mult(self, regime: str, engine_name: str) -> float:
        """
        Return multiplier untuk engine weight. 0 = disabled.
        Kalau enabled, multiplier bisa naik/turun berdasarkan accuracy.
        """
        stat = self.data.get(regime, {}).get(engine_name)
        if stat is None:
            return 1.0
        if stat.disabled:
            return 0.0
        if stat.n < self.min_samples:
            return 1.0
        acc = stat.recent_accuracy(window=30)
        # Map 0.5 -> 1.0x, 0.7 -> 1.3x, 0.3 -> 0.7x, clamp [0.4, 1.5]
        mult = 0.6 + (acc - 0.3) * 1.5
        return max(0.4, min(1.5, mult))

    def report(self) -> dict:
        out = {}
        for regime, engines in self.data.items():
            out[regime] = {}
            for name, stat in engines.items():
                out[regime][name] = {
                    "n": stat.n,
                    "accuracy": round(stat.accuracy(), 3),
                    "recent_accuracy": round(stat.recent_accuracy(), 3),
                    "disabled": stat.disabled,
                    "reason": stat.disabled_reason[:60] if stat.disabled_reason else "",
                }
        return out

    def total_samples(self) -> int:
        return sum(s.n for engines in self.data.values() for s in engines.values())
