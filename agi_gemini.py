#!/usr/bin/env python3
"""Gemini-powered debate, reflection, weekly report."""

import os
import json
import html
import datetime
from typing import Optional, List
import requests

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-1.5-flash:generateContent")


def _gemini_call(prompt: str, api_key: str, timeout: int = 25) -> Optional[str]:
    if not api_key:
        return None
    try:
        r = requests.post(
            GEMINI_URL,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json",
                     "X-goog-api-key": api_key},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            return None
        parts = (cands[0].get("content") or {}).get("parts") or []
        return parts[0].get("text", "").strip() if parts else None
    except Exception:
        return None


# =============================================================================
# MULTI-AGENT DEBATE
# =============================================================================
DEBATE_FILE = ".state_cache/debate.json"


def debate(signal: str, price: float, consensus: float, regime: str,
           h1_trend: str, h4_trend: str, rsi: float, atr: float,
           memory_stats: dict, anomaly_score: float,
           api_key: str) -> dict:
    if not api_key:
        return {"verdict": "ABSTAIN", "confidence_mult": 1.0,
                "notes": "no_key", "raw": ""}

    prompt = (
        "Kamu adalah dewan 3 trader institusional yang berdebat soal sinyal XAUUSD.\n\n"
        f"DATA:\n"
        f"- Arah: {signal}\n- Harga: {price:.2f}\n- Confluence: {consensus:.1f}%\n"
        f"- Regime: {regime}\n- H1: {h1_trend}\n- H4: {h4_trend}\n"
        f"- RSI: {rsi:.1f}\n- ATR: {atr:.2f}\n"
        f"- Memory: n={memory_stats.get('n',0)} wr={memory_stats.get('winrate',0)}%\n"
        f"- Anomaly: {anomaly_score:.2f}\n\n"
        "Berikan:\n"
        "1. Argumen BULL (1 kalimat)\n"
        "2. Argumen BEAR (1 kalimat)\n"
        "3. Argumen RISK (1 kalimat)\n"
        "4. VERDICT: AGREE / DISAGREE / ABSTAIN\n"
        "5. CONFIDENCE_MULT: 0.5 - 1.2\n"
        "6. NOTES: 1 kalimat\n"
        "Format:\nVERDICT: <...>\nCONFIDENCE_MULT: <...>\nNOTES: <...>\nRAW: <ringkas>"
    )
    text = _gemini_call(prompt, api_key)
    if not text:
        return {"verdict": "ABSTAIN", "confidence_mult": 1.0,
                "notes": "api_fail", "raw": ""}
    out = {"verdict": "ABSTAIN", "confidence_mult": 1.0, "notes": "", "raw": text}
    for line in text.splitlines():
        lu = line.strip().upper()
        if lu.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if "DISAGREE" in v:
                out["verdict"] = "DISAGREE"
            elif "AGREE" in v:
                out["verdict"] = "AGREE"
        elif lu.startswith("CONFIDENCE_MULT:"):
            try:
                val = float(line.split(":", 1)[1].strip())
                out["confidence_mult"] = max(0.5, min(1.2, val))
            except Exception:
                pass
        elif lu.startswith("NOTES:"):
            out["notes"] = line.split(":", 1)[1].strip()

    try:
        with open(DEBATE_FILE, "w") as f:
            json.dump(out, f)
    except Exception:
        pass
    return out


# =============================================================================
# REFLECTION
# =============================================================================
REFLECT_FILE = ".state_cache/reflection.json"


def reflect(journal: List[dict], regime: str, api_key: str,
            interval: int = 10) -> Optional[str]:
    if not api_key:
        return None
    evaluated = [t for t in journal if t.get("evaluated")]
    cache = {}
    if os.path.exists(REFLECT_FILE):
        try:
            with open(REFLECT_FILE) as f:
                cache = json.load(f)
        except Exception:
            pass
    last_n = cache.get("last_count", 0)
    if len(evaluated) - last_n < interval:
        return cache.get("last_reflection")

    recent = evaluated[-20:]
    if len(recent) < 5:
        return None
    wins = sum(1 for t in recent if t.get("result") == "WIN")
    losses = sum(1 for t in recent if t.get("result") == "LOSS")
    wr = wins / max(1, wins + losses) * 100

    lines = []
    for t in recent[-10:]:
        lines.append(
            f"- {t.get('signal')} @ {t.get('price')} regime={t.get('regime','?')} "
            f"grade={t.get('grade','?')} -> {t.get('result')}"
        )
    prompt = (
        "Kamu adalah Quant Coach untuk sistem XAUUSD otomatis.\n"
        f"Regime saat ini: {regime}\n"
        f"Winrate 20 trade terakhir: {wr:.1f}%\n"
        + "\n".join(lines) + "\n\n"
        "Berikan refleksi 3 poin (masing-masing 1 kalimat):\n"
        "1. Yang berhasil\n2. Yang gagal\n3. Saran 10 trade berikutnya\n"
        "Bahasa Indonesia, tanpa basa-basi."
    )
    text = _gemini_call(prompt, api_key)
    if text:
        cache["last_count"] = len(evaluated)
        cache["last_reflection"] = text
        try:
            with open(REFLECT_FILE, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass
    return text


def get_last_reflection() -> Optional[str]:
    if os.path.exists(REFLECT_FILE):
        try:
            with open(REFLECT_FILE) as f:
                return json.load(f).get("last_reflection")
        except Exception:
            pass
    return None


# =============================================================================
# WEEKLY REPORT
# =============================================================================
def build_weekly_report(journal: List[dict], api_key: str,
                        chart_path: str = ".state_cache/weekly_chart.png") -> dict:
    now = datetime.datetime.utcnow()
    start = now - datetime.timedelta(days=7)

    in_window = []
    for t in journal:
        if not t.get("evaluated"):
            continue
        try:
            dt = datetime.datetime.fromisoformat(t["time"])
            if dt.tzinfo:
                dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        except Exception:
            continue
        if start <= dt <= now:
            in_window.append(t)

    wins = [t for t in in_window if t.get("result") == "WIN"]
    losses = [t for t in in_window if t.get("result") == "LOSS"]
    rs = [t.get("pnl_r", 0) for t in in_window]
    gw = sum(r for r in rs if r > 0)
    gl = abs(sum(r for r in rs if r < 0))
    pf = gw / gl if gl > 0 else (999 if gw > 0 else 0)
    eq = 0
    peak = 0
    dd = 0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)

    # Chart
    chart = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(rs) >= 3:
            eq_curve = []
            s = 0
            for r in rs:
                s += r
                eq_curve.append(s)
            plt.figure(figsize=(10, 4))
            plt.plot(eq_curve, color="gold", linewidth=1.5)
            plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
            plt.title(f"Cumulative R — last 7 days (n={len(rs)})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(chart_path, dpi=120)
            plt.close("all")
            chart = chart_path
    except Exception:
        chart = None

    stats = {
        "period": f"{start.date()} → {now.date()}",
        "total": len(in_window),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(len(wins) / max(1, len(wins) + len(losses)) * 100, 2),
        "total_r": round(sum(rs), 2),
        "avg_r": round(sum(rs) / len(rs), 3) if rs else 0,
        "pf": round(pf, 2),
        "max_dd_r": round(dd, 2),
    }

    review = ""
    if api_key and in_window:
        prompt = (
            "Kamu adalah Quant Coach XAUUSD. Statistik mingguan:\n"
            f"Trades: {stats['total']} (W{stats['wins']}/L{stats['losses']})\n"
            f"Winrate: {stats['winrate']}% | Total R: {stats['total_r']:+.2f}\n"
            f"PF: {stats['pf']} | MaxDD: {stats['max_dd_r']:.2f}R\n\n"
            "Tulis review 3 paragraf bahasa Indonesia: (1) ringkasan performa, "
            "(2) insight regime/engine, (3) rekomendasi konkret minggu depan. "
            "Tanpa basa-basi."
        )
        review = _gemini_call(prompt, api_key, timeout=30) or ""
    if not review:
        review = (
            f"Minggu ini {stats['total']} trade dengan WR {stats['winrate']}%, "
            f"total {stats['total_r']:+.2f}R, PF {stats['pf']}."
        )

    return {"stats": stats, "review": review, "chart": chart}


def format_weekly(report: dict) -> str:
    s = report["stats"]
    rv = html.escape(report["review"][:1200])
    return (
        f"📊 <b>WEEKLY REPORT</b>\n<i>{s['period']}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Trades : {s['total']} (W{s['wins']}/L{s['losses']})\n"
        f"Winrate: {s['winrate']}%\n"
        f"Total R: {s['total_r']:+.2f}\n"
        f"Avg R  : {s['avg_r']:+.3f}\n"
        f"PF     : {s['pf']}\n"
        f"MaxDD  : {s['max_dd_r']:.2f}R\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 <b>AI Review</b>\n<i>{rv}</i>"
    )
