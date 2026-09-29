#!/usr/bin/env python3
"""Test suite for AGI core, telegram utils, main pipeline."""

import os
import sys
import json
import importlib.util
import datetime
import pytest
import numpy as np
import pandas as pd
import pytz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Isolate state dir
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""
os.environ["GEMINI_API_KEY"] = ""

import agi_core
import agi_gemini

UTC = pytz.UTC


def _df(n=300, kind="up", seed=1):
    np.random.seed(seed)
    idx = pd.date_range(end=datetime.datetime.now(UTC), periods=n,
                        freq="15min", tz="UTC")
    if kind == "up":
        c = np.linspace(1950, 2050, n) + np.random.normal(0, 1.5, n)
    elif kind == "down":
        c = np.linspace(2050, 1950, n) + np.random.normal(0, 1.5, n)
    else:
        c = 2000 + np.random.normal(0, 3, n)
    return pd.DataFrame({
        "open": c + np.random.normal(0, 0.5, n),
        "high": c + np.abs(np.random.normal(0, 2, n)),
        "low": c - np.abs(np.random.normal(0, 2, n)),
        "close": c,
        "volume": np.abs(np.random.normal(100, 15, n)),
    }, index=idx)


# =============================================================================
class TestRegime:
    def test_returns_regime_state(self):
        rs = agi_core.detect_regime(_df(300, "up"))
        assert 0 <= rs.confidence <= 1
        assert rs.regime in list(agi_core.Regime)

    def test_insufficient(self):
        rs = agi_core.detect_regime(_df(30))
        assert rs.regime == agi_core.Regime.TRANSITION

    def test_none(self):
        rs = agi_core.detect_regime(None)
        assert rs.regime == agi_core.Regime.TRANSITION


class TestAnomaly:
    def test_normal_low(self):
        rep = agi_core.detect_anomaly(_df(300, "up", seed=5))
        assert rep.score < 0.7

    def test_shock_high(self):
        df = _df(300, "up", seed=6)
        df.iloc[-1, df.columns.get_loc("close")] *= 1.05
        df.iloc[-1, df.columns.get_loc("high")] *= 1.08
        df.iloc[-1, df.columns.get_loc("volume")] *= 10
        rep = agi_core.detect_anomaly(df)
        assert rep.score > 0.3


class TestEmbedding:
    def test_shape(self):
        v = agi_core.encode(_df(300))
        assert v.shape == (agi_core.EMB_DIM,)
        assert abs(np.linalg.norm(v) - 1) < 0.01

    def test_empty(self):
        v = agi_core.encode(pd.DataFrame())
        assert np.all(v == 0)

    def test_similar_deterministic(self):
        v1 = agi_core.encode(_df(300, "up", seed=10))
        v2 = agi_core.encode(_df(300, "up", seed=10))
        assert np.allclose(v1, v2)


class TestMemory:
    def test_store_query(self, tmp_path):
        m = agi_core.Memory(path=str(tmp_path / "m.db"))
        df = _df(300, "up")
        rs = agi_core.detect_regime(df)
        for _ in range(10):
            m.store(df, rs, "BUY", 2000, "WIN", 1.5)
        entries, stats = m.query(df, k=5)
        assert stats["n"] > 0
        assert stats["winrate"] > 0

    def test_empty(self, tmp_path):
        m = agi_core.Memory(path=str(tmp_path / "m.db"))
        entries, stats = m.query(_df(300), k=5)
        assert stats["n"] == 0


class TestCalibration:
    def test_unfitted(self, tmp_path):
        c = agi_core.Calibrator(path=str(tmp_path / "c.json"))
        assert c.calibrate(0.7) == 0.7

    def test_fit(self, tmp_path):
        c = agi_core.Calibrator(path=str(tmp_path / "c.json"))
        for _ in range(30):
            c.accumulate(0.8, 1)
            c.accumulate(0.3, 0)
        c.fit_from_samples()
        assert c.fitted is True
        assert c.calibrate(0.9) > c.calibrate(0.3)


class TestMeta:
    def test_default(self, tmp_path):
        m = agi_core.MetaLearner(path=str(tmp_path / "m.json"))
        assert m.penalty("TRENDING_UP") == 1.0

    def test_losses_drop(self, tmp_path):
        m = agi_core.MetaLearner(path=str(tmp_path / "m.json"), alpha=0.3)
        for _ in range(20):
            m.update("RANGING", False, -1.0)
        assert m.penalty("RANGING") < 1.0

    def test_wins_rise(self, tmp_path):
        m = agi_core.MetaLearner(path=str(tmp_path / "m.json"), alpha=0.3)
        for _ in range(30):
            m.update("RANGING", True, 1.5)
        assert m.penalty("RANGING") > 1.0


class TestGrade:
    def test_grade_high(self):
        states = {f"E{i}": {"sc": 1} for i in range(10)}
        r = agi_core.grade_signal(85, "BUY", states,
                                    "BULLISH (UP)", "BULLISH (UP)", 8,
                                    {"n": 20, "winrate": 65, "avg_r": 0.5}, 0.1)
        assert r["grade"] in ("A++", "A+++")

    def test_grade_low(self):
        states = {f"E{i}": {"sc": -1 if i % 2 else 1} for i in range(10)}
        r = agi_core.grade_signal(55, "BUY", states,
                                    "BEARISH (DOWN)", "NEUTRAL", 20,
                                    {"n": 5, "winrate": 35, "avg_r": -0.2}, 0.8)
        assert r["grade"] in ("C", "D")

    def test_at_least(self):
        assert agi_core.grade_at_least("A", "A") is True
        assert agi_core.grade_at_least("A+++", "A") is True
        assert agi_core.grade_at_least("B", "A") is False


class TestPersonality:
    def test_caption(self):
        rs = agi_core.RegimeState(agi_core.Regime.TRENDING_UP, 0.8, 30,
                                    0.5, 0.5, 0.01, "")
        an = agi_core.AnomalyReport(0.1, False, "normal")
        cap = agi_core.compose_caption(
            "BUY", 2000, 1990, [2010, 2020, 2030, 2040], 8.0,
            75, "A", 85, rs, an, {"n": 10, "winrate": 60, "avg_r": 0.5},
            "AGREE", "Solid setup", "Deriv", 0, "Insight",
        )
        assert "BUY" in cap
        assert "2000" in cap
        assert "A" in cap


class TestGeminiParsing:
    def test_no_key_returns_abstain(self):
        r = agi_gemini.debate("BUY", 2000, 75, "TRENDING_UP", "BULLISH", "NEUTRAL",
                               60, 8, {"n": 5, "winrate": 60}, 0.1, "")
        assert r["verdict"] == "ABSTAIN"

    def test_weekly_empty(self):
        r = agi_gemini.build_weekly_report([], api_key="")
        assert r["stats"]["total"] == 0
        assert "idle" in r["review"].lower() or "trade" in r["review"].lower()


class TestPipelineEval:
    def test_evaluate_trade_win(self):
        # Import main pipeline
        spec = importlib.util.spec_from_file_location(
            "quant_engine", os.path.join(ROOT, "quant_engine.py"))
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)

        now = datetime.datetime.now(UTC)
        trade = {
            "time": (now - datetime.timedelta(minutes=45)).isoformat(),
            "signal": "BUY", "price": 2000, "sl": 1990, "tp1": 2020,
        }
        idx = pd.date_range(start=now - datetime.timedelta(minutes=40),
                            periods=3, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "open": [2005, 2010, 2020], "high": [2010, 2021, 2025],
            "low": [2000, 2005, 2015], "close": [2005, 2015, 2022],
            "volume": [100, 100, 100],
        }, index=idx)
        res, ch = q.evaluate_trade(trade, df)
        assert res == "WIN"

    def test_evaluate_trade_loss(self):
        spec = importlib.util.spec_from_file_location(
            "quant_engine", os.path.join(ROOT, "quant_engine.py"))
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)

        now = datetime.datetime.now(UTC)
        trade = {
            "time": (now - datetime.timedelta(minutes=45)).isoformat(),
            "signal": "BUY", "price": 2000, "sl": 1990, "tp1": 2020,
        }
        idx = pd.date_range(start=now - datetime.timedelta(minutes=40),
                            periods=3, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "open": [1995, 1990, 1985], "high": [1998, 1992, 1988],
            "low": [1988, 1985, 1980], "close": [1990, 1986, 1982],
            "volume": [100, 100, 100],
        }, index=idx)
        res, ch = q.evaluate_trade(trade, df)
        assert res == "LOSS"


class TestEngines:
    def test_all_engines(self):
        spec = importlib.util.spec_from_file_location(
            "quant_engine", os.path.join(ROOT, "quant_engine.py"))
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)

        df = _df(300, "up")
        for name, fn in q.ENGINES:
            sc, w = fn(df)
            assert sc in (-1, 0, 1)
            assert w > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
