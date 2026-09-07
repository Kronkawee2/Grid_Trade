"""
Take the winner from the training search and run it once on 2025-2026.

This is the only time those years are used. The value of an out-of-sample
period is spent by looking at it: every peek lets a choice be made with
knowledge of the answer, and after enough peeks it is training data
wearing a different label. So this script does not search, does not offer
variants, and reports whatever comes out.

    python scripts/backtest/test_oos.py

Writes best_params.json (so the plotting script runs the identical
configuration) and the out-of-sample chart.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402
from plot_run import draw  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SYMBOL = "EURUSD.DUKA"
TRAIN_START, TRAIN_END = "2006-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2026-12-31"
SPREAD_PIPS = 1.0

PARAM_KEYS = ["atr_mult", "max_open", "stop_extra", "vol_block",
              "atr_fast", "atr_slow", "side", "max_hold_days",
              "er_window", "er_block", "dd_pause", "dd_pause_bars"]
# `side` is no longer searched; it is pinned to a direction-neutral grid.
DEFAULTS = {"side": "both", "er_window": 0, "er_block": 1.0,
            "dd_pause": None, "dd_pause_bars": 0}
INT_KEYS = {"max_open", "atr_fast", "atr_slow", "max_hold_days",
            "er_window", "dd_pause_bars"}


def load_best(path: Path) -> dict:
    df = pd.read_csv(path)
    df = df[np.isfinite(df["score"])]
    best = df.loc[df["score"].idxmax()]
    params = {}
    for k in PARAM_KEYS:
        if k not in best.index:
            params[k] = DEFAULTS[k]
            continue
        v = best[k]
        # `max_hold_days=None` survives the CSV round trip as NaN, and a
        # NaN handed back to the engine would silently disable the time
        # stop while the report still claimed a value.
        if pd.isna(v):
            params[k] = None
        elif k in INT_KEYS:
            params[k] = int(v)
        elif isinstance(v, str):
            params[k] = v
        else:
            params[k] = float(v)
    return params, best


def report(name, res, cfg, start_cash):
    eq = res["equity"]
    daily = eq.resample("1D").last().dropna().diff().dropna()
    dd = abs(res["max_drawdown"])
    print(f"\n--- {name} ---")
    print(f"  net profit      : ${res['net_profit']:,.0f}")
    print(f"  per trading day : ${res['per_day']:.2f}")
    print(f"  max drawdown    : ${res['max_drawdown']:,.0f}  "
          f"({dd / start_cash * 100:.0f}% of ${start_cash:,.0f})")
    print(f"  net / drawdown  : {res['net_profit'] / dd:.2f}" if dd else "")
    print(f"  trades          : {res['n_trades']:,}   win {res['win_rate']:.1f}%")
    print(f"  stops           : {res['n_stops']:,}")
    print(f"  peak open       : {res['max_concurrent']} (cap {cfg.max_open})")
    print(f"  days blocked    : {res['n_blocked_bars']:,} bars stood aside")
    print(f"  worst day       : ${daily.min():,.2f}   best day ${daily.max():,.2f}")
    return dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="m5")
    ap.add_argument("--results", default=None)
    ap.add_argument("--cash", type=float, default=100.0)
    ap.add_argument("--target-per-day", type=float, default=10.0)
    args = ap.parse_args()

    TIMEFRAME = args.timeframe
    results = Path(args.results) if args.results else (
        ROOT / f"results_optimize_{TIMEFRAME}.csv")
    params, best_row = load_best(results)
    (ROOT / f"best_params_{TIMEFRAME}.json").write_text(json.dumps(params, indent=2))
    print("configuration chosen on 2006-2024, unchanged from here on:")
    for k, v in params.items():
        print(f"  {k:14}: {v}")

    tr_df = load_bars(SYMBOL, TIMEFRAME, start=TRAIN_START, end=TRAIN_END)
    oos_df = load_bars(SYMBOL, TIMEFRAME, start=OOS_START, end=OOS_END)
    print(f"\ntrain {len(tr_df):,} bars {tr_df.index[0]:%Y-%m-%d} -> {tr_df.index[-1]:%Y-%m-%d}")
    print(f"test  {len(oos_df):,} bars {oos_df.index[0]:%Y-%m-%d} -> {oos_df.index[-1]:%Y-%m-%d}")

    cfg = AdaptiveConfig(spread_pips=SPREAD_PIPS, **params)
    tr_res = run_adaptive(tr_df, cfg)
    oos_res = run_adaptive(oos_df, cfg)

    report("training 2006-2024 (the search saw this)", tr_res, cfg, args.cash)
    oos_dd = report("OUT OF SAMPLE 2025-2026 (never seen)", oos_res, cfg, args.cash)

    # How much of the training edge survived
    tr_pd, oos_pd = tr_res["per_day"], oos_res["per_day"]
    print("\n--- did it survive ---")
    print(f"  per day, train : ${tr_pd:.2f}")
    print(f"  per day, test  : ${oos_pd:.2f}"
          f"   ({oos_pd / tr_pd * 100:.0f}% of training)" if tr_pd else "")

    # What the user's target implies, at this risk shape
    if oos_pd > 0:
        scale = args.target_per_day / oos_pd
        print(f"\n--- what ${args.target_per_day:.0f}/day would take ---")
        print(f"  lot size needed : {cfg.lot * scale:.2f} "
              f"(from {cfg.lot:.2f}, x{scale:.0f})")
        print(f"  drawdown scales identically: ${oos_res['max_drawdown'] * scale:,.0f}")
        print(f"  capital to hold that at 20% risk: "
              f"${abs(oos_res['max_drawdown']) * scale * 5:,.0f}")
    else:
        print(f"\n--- what ${args.target_per_day:.0f}/day would take ---")
        print("  out-of-sample per-day is not positive, so no lot size reaches "
              "the target. Scaling a negative expectancy only scales the loss.")

    png = ROOT / "plots" / f"oos_2025_2026_{TIMEFRAME}.png"
    draw(oos_df, oos_res, cfg,
         f"{SYMBOL} {TIMEFRAME} adaptive grid, OUT OF SAMPLE {OOS_START} -> "
         f"{oos_df.index[-1]:%Y-%m-%d}  |  net ${oos_res['net_profit']:,.0f}  "
         f"maxDD ${oos_res['max_drawdown']:,.0f}  ${oos_res['per_day']:.2f}/day  "
         f"{oos_res['n_trades']:,} trades  max {oos_res['max_concurrent']} open",
         png)
    print(f"\nsaved -> {ROOT / 'best_params.json'}")
    print(f"saved -> {png}")


if __name__ == "__main__":
    main()
