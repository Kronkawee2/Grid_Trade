"""
Walk-forward: does last year's best grid still work next year?

The question this answers is not "what are the best parameters" -- any
search returns something for that, and the number it returns is usually
the luckiest point rather than the best one. The question is whether a
choice made with no sight of the future survives contact with it.

So the search is run inside a window, the winner is applied to the year
that follows, and only that following year is scored. Repeat, sliding
forward. Nothing that scores a year was fitted on it.

Reading the output
------------------
Three outcomes, and they lead to different projects:

  the winning parameters jump around every year, and out-of-sample
  results are noise      -> the strategy has no edge and no model will
                            create one; stop here

  the winning parameters cluster, and out-of-sample is consistently
  positive               -> a fixed configuration is enough; ML would be
                            complexity for its own sake

  the winning parameters move, but move *with* something measurable such
  as volatility          -> this is the case that justifies ML, and it
                            names the job: predict next period's setting,
                            not next period's price

    python scripts/backtest/walkforward.py
    python scripts/backtest/walkforward.py --train-years 3 --test-years 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import GridConfig, run_grid, summarise  # noqa: E402
from run_baseline import (  # noqa: E402
    PIP, PIP_VALUE, SPREAD_PIPS, SWAP_LONG, SWAP_SHORT, SYMBOL, TIMEFRAME,
    TRAIN_END, TRAIN_START,
)

from quantdata import load_bars  # noqa: E402

# Deliberately coarse. A fine search over this many windows would mostly
# measure how many combinations were tried, and the spacing values here
# already straddle EURUSD's 57-pip median daily range.
SPACINGS = [20.0, 30.0, 42.0, 60.0, 80.0]
LEVELS = [5, 10, 20]
SIDES = ["buy", "sell", "both"]

# A window that produced almost no trading tells us nothing about the
# parameters, only that price sat still. Scoring it would let a dead
# configuration win a window by never risking anything.
MIN_TRADES = 20


def score(res, summary):
    """
    Return per unit of drawdown, which is the only currency a $100 account
    has. Raw profit would always pick the widest grid holding the deepest
    losses, because on a rising sample that looks free.
    """
    dd = abs(res["max_drawdown"])
    if res["n_trades"] < MIN_TRADES or dd == 0:
        return -np.inf
    return res["net_profit"] / dd


def evaluate(df, spacing, levels, side):
    cfg = GridConfig(spacing_pips=spacing, n_levels=levels, side=side, pip=PIP,
                     pip_value=PIP_VALUE, swap_long_pips=SWAP_LONG,
                     swap_short_pips=SWAP_SHORT, spread_override_pips=SPREAD_PIPS)
    res = run_grid(df, cfg)
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    return res, summarise(res, years)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-years", type=int, default=2)
    ap.add_argument("--test-years", type=int, default=1)
    args = ap.parse_args()

    df = load_bars(SYMBOL, TIMEFRAME, start=TRAIN_START, end=TRAIN_END)
    print(f"{SYMBOL} {TIMEFRAME}: {len(df):,} bars, "
          f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    print(f"search: {len(SPACINGS)}x{len(LEVELS)}x{len(SIDES)} = "
          f"{len(SPACINGS)*len(LEVELS)*len(SIDES)} configs per window")
    print(f"windows: train {args.train_years}y -> test {args.test_years}y, "
          f"spread {SPREAD_PIPS:.1f}p flat\n")

    first, last = df.index[0].year, df.index[-1].year
    rows = []

    test_start_year = first + args.train_years
    while test_start_year + args.test_years - 1 <= last:
        tr = df.loc[f"{test_start_year - args.train_years}":f"{test_start_year - 1}"]
        te = df.loc[f"{test_start_year}":f"{test_start_year + args.test_years - 1}"]
        if len(tr) < 5000 or len(te) < 5000:
            test_start_year += args.test_years
            continue

        best = None
        for spacing in SPACINGS:
            for levels in LEVELS:
                for side in SIDES:
                    res, s = evaluate(tr, spacing, levels, side)
                    sc = score(res, s)
                    if best is None or sc > best[0]:
                        best = (sc, spacing, levels, side, res, s)

        sc, spacing, levels, side, tr_res, tr_s = best
        te_res, te_s = evaluate(te, spacing, levels, side)

        rows.append({
            "test_year": test_start_year,
            "spacing": spacing, "levels": levels, "side": side,
            "in_score": sc, "in_net": tr_res["net_profit"], "in_dd": tr_res["max_drawdown"],
            "out_score": score(te_res, te_s),
            "out_net": te_res["net_profit"], "out_dd": te_res["max_drawdown"],
            "out_trades": te_res["n_trades"], "out_stops": te_res["n_stops"],
        })
        r = rows[-1]
        print(f"  test {r['test_year']} | chose {spacing:>4.0f}p x{levels:<3} {side:<5} "
              f"| in ${r['in_net']:>8,.0f} / DD ${r['in_dd']:>8,.0f} "
              f"|| OUT ${r['out_net']:>8,.0f} / DD ${r['out_dd']:>8,.0f} "
              f"| {r['out_trades']:>5,} trades")

        test_start_year += args.test_years

    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parent.parent.parent / "results_walkforward.csv"
    out.to_csv(path, index=False)

    print("\n--- out-of-sample, aggregated ---")
    pos = (out["out_net"] > 0).sum()
    print(f"profitable years      : {pos}/{len(out)}")
    print(f"total out-of-sample   : ${out['out_net'].sum():,.0f}")
    print(f"median year           : ${out['out_net'].median():,.0f}")
    print(f"worst year            : ${out['out_net'].min():,.0f} "
          f"(deepest DD ${out['out_dd'].min():,.0f})")
    print("\nparameter stability (how often each choice won):")
    for col in ("spacing", "levels", "side"):
        counts = out[col].value_counts().sort_index()
        print(f"  {col:8}: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
