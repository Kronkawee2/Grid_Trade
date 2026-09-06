"""
Gate 2 and 3: is the baseline still on a plateau, on the engine in use?

Gate 2 -- the 45-cell scan that justified the current values ran before
the equity stop existed, and without the equity stop both markets are
ruined. That plateau therefore describes a different system.

Gate 3 -- two parameters were frozen through every scan and never tested.
One of them matters: atr_fast=10 returns $877 at a 16% drawdown where the
current 20 returns $594 at 36%.

So the scan is repeated on the current engine with those two included.
The baseline sits at the centre of every axis, which makes the question
precise: are its neighbours still good, and is the centre still the right
place to stand?

EURUSD only. Gold takes no part in choosing anything.

    python scripts/backtest/gate23_plateau.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
SYMBOL, SPEC = "EURUSD.DUKA", "EURUSD"
START, END, CASH = "2006-01-01", "2024-12-31", 100.0
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
BASE = json.loads((ROOT / "params_100usd.json").read_text())

# The baseline value sits in the middle of each list.
AXES = {
    "atr_mult":  [3.5, 4.2, 5.0],
    "rebound":   [0.35, 0.45, 0.55],
    "vol_block": [1.00, 1.15, 1.35],
    "atr_fast":  [10, 20, 50],        # never tested before
    "atr_slow":  [100, 200, 288],     # never tested before; 288 = one day
}
HELD = dict(stop_extra=BASE["stop_extra"], max_open=BASE["max_open"],
            equity_stop=BASE["equity_stop"], dd_pause_bars=BASE["dd_pause_bars"],
            side="both", vol_floor=0.0, er_window=0, er_block=1.0,
            dd_pause=None, max_hold_days=None, block_hours=())


def money():
    s = SPECS[SPEC]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=40.0)


def measure(df, p, years):
    res = run_adaptive(df, AdaptiveConfig(**p, **HELD, **money()))
    eq = CASH + res["equity"]
    peak = eq.cummax()
    dd = float(((peak - eq) / peak).max())
    monthly = res["equity"].resample("ME").last().dropna().diff().dropna()
    ye = eq.resample("YE").last()
    ypct = (ye / ye.shift(1).fillna(CASH) - 1) * 100
    return {**p, "final": round(float(eq.iloc[-1]), 2),
            "max_dd_pct": round(dd * 100, 2),
            "months_up_pct": round(float((monthly > 0).mean()) * 100, 1),
            "median_year_pct": round(float(ypct.median()), 2),
            "years_up": int((ypct > 0).sum()),
            "trades_per_year": round(res["n_trades"] / years, 0),
            "ruined": bool(eq.min() <= 0)}


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_bars(SYMBOL, "m5", start=START, end=END)
    years = (df.index[-1] - df.index[0]).days / 365.25
    combos = list(itertools.product(*AXES.values()))
    print(f"{SYMBOL} m5, {len(df):,} bars. {len(combos)} cells "
          f"({' x '.join(str(len(v)) for v in AXES.values())}), EURUSD only\n",
          flush=True)

    rows, t0 = [], time.time()
    for i, c in enumerate(combos, 1):
        rows.append(measure(df, dict(zip(AXES, c)), years))
        if i % 27 == 0 or i == len(combos):
            el = time.time() - t0
            print(f"  {i:>4}/{len(combos)}  {el/i:.1f}s each  "
                  f"eta {(len(combos)-i)*el/i/60:.0f}m", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "gate23_plateau.csv", index=False)

    base = {k: AXES[k][1] for k in AXES}
    b = out[np.logical_and.reduce([out[k] == v for k, v in base.items()])].iloc[0]

    print(f"\n{'='*70}\nGATE 2 -- is the baseline still on a plateau?\n{'='*70}")
    print(f"  cells that made money   : {(out['final'] > CASH).sum()}/{len(out)}")
    print(f"  cells that ruined the account: {int(out['ruined'].sum())}")
    print(f"  final balance   median ${out['final'].median():,.0f}   "
          f"worst ${out['final'].min():,.0f}   best ${out['final'].max():,.0f}")
    verdict2 = (out["final"] > CASH).all()
    print(f"  -> {'PASS' if verdict2 else 'FAIL'}  "
          f"(a plateau needs every neighbour still profitable)")

    print(f"\n{'='*70}\nGATE 3 -- is the baseline the best place to stand?\n{'='*70}")
    print(f"  baseline: ${b['final']:,.0f}  dd {b['max_dd_pct']}%  "
          f"months {b['months_up_pct']}%  median year {b['median_year_pct']}%")
    for crit, better in (("final", "greater"), ("max_dd_pct", "less"),
                         ("months_up_pct", "greater"), ("median_year_pct", "greater")):
        rank = int((out[crit] > b[crit]).sum() if better == "greater"
                   else (out[crit] < b[crit]).sum()) + 1
        print(f"  ranks {rank:>3} of {len(out)} on {crit}")
    verdict3 = int((out["final"] > b["final"]).sum()) <= len(out) * 0.1
    print(f"  -> {'PASS' if verdict3 else 'FAIL'}  "
          f"(needs to be in the top 10% of its own neighbourhood)")

    print("\ntop 8 cells by final balance:")
    cols = list(AXES) + ["final", "max_dd_pct", "months_up_pct",
                         "median_year_pct", "trades_per_year"]
    print(out.nlargest(8, "final")[cols].to_string(index=False))

    print("\neach axis, averaged over everything else:")
    for k in AXES:
        g = out.groupby(k).agg(final=("final", "median"),
                               dd=("max_dd_pct", "median"),
                               months=("months_up_pct", "median")).round(1)
        print(f"\n  {k}:")
        print(g.to_string())

    print(f"\nsaved -> {RESULTS / 'gate23_plateau.csv'}")


if __name__ == "__main__":
    main()
