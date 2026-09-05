"""
Baseline run: a plain fixed grid, three side configurations, training data only.

The out-of-sample period (2025 onward) is not touched here. It exists to
answer one question once, at the end of the research, and every look at it
before then spends a little of that answer.

    python scripts/backtest/run_baseline.py
    python scripts/backtest/run_baseline.py --spacing 20 40 60 --levels 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import GridConfig, run_grid, summarise  # noqa: E402

from quantdata import load_bars  # noqa: E402

SYMBOL = "EURUSD.DUKA"
TIMEFRAME = "m15"

# Fixed here rather than passed in, so that no run can quietly slide the
# boundary after seeing a result.
TRAIN_START = "2006-01-01"
TRAIN_END = "2024-12-31"

# EURUSD on this broker: 0.01 lot moves $0.10 per pip; longs pay 0.7 pip a
# night and shorts collect 0.25. Both read live from MT5 and they match.
PIP = 0.0001
PIP_VALUE = 1.0
SWAP_LONG = -0.7
SWAP_SHORT = 0.25

# Eightcap's live EURUSD spread, held constant across the whole history.
#
# The recorded spread in this dataset is Dukascopy's, and Dukascopy is an
# ECN quoting roughly a third of what this retail account is charged --
# 0.32 pip in 2024 against Eightcap's 1.0. The prices agree to a fraction
# of a pip, but the spread is the broker's fee, not the market's, so it
# does not transfer. Nor is there an Eightcap spread history to scale
# against: those tables carry no spread column at all.
#
# A constant therefore misses the widening around news, which is exactly
# when a grid is filling. That understates cost, so treat every result
# here as the optimistic end. It is a five-minute change if a better cost
# model is wanted later.
SPREAD_PIPS = 1.0


def main():
    ap = argparse.ArgumentParser(description="Baseline fixed-grid backtest")
    ap.add_argument("--spacing", type=float, nargs="+", default=[42.0],
                    help="grid spacing in pips")
    ap.add_argument("--levels", type=int, nargs="+", default=[10])
    ap.add_argument("--stop", type=float, default=None,
                    help="stop distance in pips (default: one full zone beyond)")
    ap.add_argument("--sides", nargs="+", default=["buy", "sell", "both"])
    ap.add_argument("--cash", type=float, default=100.0)
    args = ap.parse_args()

    df = load_bars(SYMBOL, TIMEFRAME, start=TRAIN_START, end=TRAIN_END)
    years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"{SYMBOL} {TIMEFRAME}: {len(df):,} bars, "
          f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} ({years:.1f}y)")
    if "spread_avg" in df.columns:
        sp = df["spread_avg"].dropna() / 10
        print(f"spread from data: median {sp.median():.2f}p, "
              f"p95 {sp.quantile(0.95):.2f}p, max {sp.max():.1f}p")
    print()

    rows = []
    for spacing in args.spacing:
        for levels in args.levels:
            for side in args.sides:
                cfg = GridConfig(spacing_pips=spacing, n_levels=levels,
                                 stop_pips=args.stop, side=side, pip=PIP,
                                 pip_value=PIP_VALUE, swap_long_pips=SWAP_LONG,
                                 swap_short_pips=SWAP_SHORT,
                                 spread_override_pips=SPREAD_PIPS)
                res = run_grid(df, cfg)
                s = summarise(res, years, args.cash)
                s.update(spacing=spacing, levels=levels, side=side,
                         zone=cfg.zone_pips, stop=cfg.stop / PIP)
                rows.append(s)
                print(f"  spacing={spacing:>5.0f}p levels={levels:>3} side={side:<5} "
                      f"| net ${s['net_profit']:>10,.0f} | maxDD ${s['max_drawdown']:>9,.0f} "
                      f"| trades {s['trades']:>8,} | stops/yr {s['stops_per_year']:>6.1f} "
                      f"| win {s['win_rate']:>5.1f}%")

    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parent.parent.parent / "results_baseline.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
