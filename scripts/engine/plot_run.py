"""
Draw one run the way the reference system draws it: price with the grid
boxes and every entry and exit marked, volatility regime underneath, and
the equity and drawdown below that.

The two extra panels are not decoration. The top panel alone can make any
run look competent -- markers cluster where price wiggled and the eye
supplies a story. The equity panel says whether the story ended with
money, and the drawdown panel says what it cost to sit through. A grid is
judged by the third panel more than the first.

    python scripts/backtest/plot_run.py --start 2025-01-01 --end 2026-09-01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SYMBOL = "EURUSD.DUKA"


def draw(df, res, cfg, title, path, max_markers=400):
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 11), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1, 1.6], "hspace": 0.07})
    ax_p, ax_v, ax_e = axes

    # --- price, grid boxes, fills ------------------------------------
    ax_p.plot(df.index, df["close"], lw=0.7, color="0.35", label="Close price")

    boxes = res["boxes"]
    # Boxes that lasted only a bar or two would paint the panel solid and
    # hide the price, so only those that actually framed a period are drawn.
    span = pd.to_datetime(boxes["end"]) - pd.to_datetime(boxes["start"])
    for _, b in boxes[span > pd.Timedelta("2D")].iterrows():
        half = b["spacing"] * b["levels"]
        ax_p.add_patch(Rectangle(
            (b["start"], b["anchor"] - half), b["end"] - b["start"], 2 * half,
            fill=False, edgecolor="#6a5acd", lw=0.8, alpha=0.55))

    tr = res["trades"]
    if len(tr):
        step = max(1, len(tr) // max_markers)
        t = tr.iloc[::step]
        styles = [
            (t[(t.side == "buy")], "opened_at", "entry", "^", "#22b14c", "Long entry"),
            (t[(t.side == "buy")], "closed_at", "exit", "v", "#e06c3a", "Long exit"),
            (t[(t.side == "sell")], "opened_at", "entry", "v", "#ff44aa", "Short entry"),
            (t[(t.side == "sell")], "closed_at", "exit", "^", "#22d3ee", "Short exit"),
        ]
        for sub, tcol, pcol, marker, colour, label in styles:
            if len(sub):
                ax_p.scatter(sub[tcol], sub[pcol], marker=marker, s=26,
                             color=colour, label=label, zorder=4, linewidths=0)
    ax_p.set_ylabel("price")
    ax_p.legend(loc="upper left", ncol=5, fontsize=8, framealpha=0.9)
    ax_p.set_title(title, fontsize=12)
    ax_p.grid(alpha=0.2)

    # --- volatility regime -------------------------------------------
    ax_v.plot(df.index, res["atr_pips"], lw=0.6, color="#3b5bdb", label="ATR (pips)")
    base = res["atr_pips"] / res["vol_ratio"].replace(0, np.nan)
    ax_v.plot(df.index, base, lw=1.0, color="orange", label="Baseline")
    blocked = res["vol_ratio"] > cfg.vol_block
    ax_v.fill_between(df.index, 0, res["atr_pips"].max(), where=blocked,
                      color="crimson", alpha=0.10, step="mid",
                      label=f"blocked (ratio > {cfg.vol_block:.2f})")
    ax_v.set_ylabel("volatility")
    ax_v.legend(loc="upper left", ncol=3, fontsize=8)
    ax_v.grid(alpha=0.2)

    # --- equity and drawdown -----------------------------------------
    ax_e.plot(df.index, res["equity"], lw=1.1, color="#111", label="Equity ($)")
    ax_e.axhline(0, color="grey", lw=0.8, ls="--")
    ax_e.fill_between(df.index, res["drawdown"], 0, color="crimson",
                      alpha=0.35, label="Drawdown ($)")
    ax_e.set_ylabel("account ($)")
    ax_e.legend(loc="upper left", ncol=2, fontsize=8)
    ax_e.grid(alpha=0.2)

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--params", default=str(ROOT / "best_params.json"),
                    help="JSON written by test_oos.py / optimize.py")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--timeframe", default="m5")
    ap.add_argument("--markers", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    params = json.loads(Path(args.params).read_text())
    params["lot"] = args.lot
    TIMEFRAME = args.timeframe
    df = load_bars(SYMBOL, TIMEFRAME, start=args.start, end=args.end)
    cfg = AdaptiveConfig(spread_pips=1.0, **params)
    res = run_adaptive(df, cfg)

    title = (f"{SYMBOL} {TIMEFRAME} adaptive grid  |  {args.start} -> {args.end}  |  "
             f"net ${res['net_profit']:,.0f}  maxDD ${res['max_drawdown']:,.0f}  "
             f"${res['per_day']:.2f}/day  {res['n_trades']:,} trades  "
             f"max {res['max_concurrent']} open")
    out = Path(args.out) if args.out else (
        ROOT / "plots" / f"run_{TIMEFRAME}_{args.start[:4]}_{args.end[:4]}.png")
    draw(df, res, cfg, title, out, max_markers=args.markers)
    print(title)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
