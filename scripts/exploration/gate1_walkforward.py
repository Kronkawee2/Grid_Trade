"""
Gate 1: choose the parameters again, sixteen times, never seeing the future.

Every number reported so far comes from values picked while looking at the
whole of 2006-2024, and then measured on that same stretch. That cannot
distinguish a configuration that works from one that fits.

This repeats the choice the way it would actually have to be made. Fit on
a training window, trade the year that follows, move forward, repeat. The
year being scored has not happened yet at the moment its parameters are
chosen, so nothing in it can leak backwards.

What matters in the output is not the profit. It is whether the winning
values stay in the same region from window to window. Values that jump
around are noise wearing a costume; values that hold, or that move with
something measurable, are the thing worth building on.

EURUSD only. Gold is never used to choose anything.

    python scripts/backtest/gate1_walkforward.py
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
SYMBOL, SPEC = "EURUSD.DUKA", "EURUSD"
CASH = 100.0
TRAIN_YEARS, TEST_YEARS = 3, 1
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
BASE = json.loads((ROOT / "params_100usd.json").read_text())

# Deliberately coarse. A fine grid inside each window would mostly measure
# how many combinations were tried, and the point here is the stability of
# the choice, not its precision.
# The current values sit in the middle of each list, so the question is
# precise: with no sight of the year being traded, does the search keep
# landing where the baseline already is?
AXES = {"atr_mult": [4.2, 5.0, 5.8],
        "rebound": [0.5, 0.65, 0.8],
        "vol_block": [0.8, 0.9, 1.05],
        "atr_fast": [5, 10, 20]}
HELD = dict(stop_extra=BASE["stop_extra"], max_open=BASE["max_open"],
            equity_stop=BASE["equity_stop"], dd_pause_bars=BASE["dd_pause_bars"],
            atr_slow=BASE["atr_slow"],
            side="both", vol_floor=0.0, er_window=0, er_block=1.0,
            dd_pause=None, max_hold_days=None, block_hours=())
MIN_TRADES = 40          # per window; below this the score is an accident


def money():
    s = SPECS[SPEC]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=40.0)


def score(df, p):
    """Return per unit of drawdown -- the account cannot spend what it did
    not survive, so profit alone is the wrong thing to rank by."""
    r = run_adaptive(df, AdaptiveConfig(**p, **HELD, **money()))
    eq = CASH + r["equity"]
    peak = eq.cummax()
    dd = float(((peak - eq) / peak).max())
    if r["n_trades"] < MIN_TRADES or dd <= 0 or eq.min() <= 0:
        return -np.inf, r, dd
    return r["net_profit"] / (dd * CASH), r, dd


def main():
    RESULTS.mkdir(exist_ok=True)
    full = load_bars(SYMBOL, "m5", start="2006-01-01", end="2024-12-31")
    combos = [dict(zip(AXES, c)) for c in itertools.product(*AXES.values())]
    years = sorted(full.index.year.unique())
    windows = [(y - TRAIN_YEARS, y) for y in years
               if y - TRAIN_YEARS >= years[0] and y <= years[-1]]
    print(f"{SYMBOL} m5, {len(combos)} configs per window, {len(windows)} windows "
          f"({TRAIN_YEARS}y train -> {TEST_YEARS}y test), EURUSD only\n", flush=True)

    rows, t0 = [], time.time()
    for n, (a, y) in enumerate(windows, 1):
        tr = full.loc[f"{a}":f"{y-1}"]
        te = full.loc[f"{y}":f"{y}"]
        if len(tr) < 50000 or len(te) < 20000:
            continue
        best, bp = -np.inf, None
        for p in combos:
            s, _, _ = score(tr, p)
            if s > best:
                best, bp = s, p
        _, r, dd = score(te, bp)
        eq = CASH + r["equity"]
        # what the frozen baseline would have done in the same year
        _, rb, ddb = score(te, {k: BASE[k] for k in AXES})
        eqb = CASH + rb["equity"]
        rows.append(dict(test_year=y, **bp, train_score=round(best, 2),
                         oos_net=round(r["net_profit"], 2),
                         oos_dd_pct=round(dd * 100, 1),
                         oos_trades=r["n_trades"],
                         baseline_net=round(rb["net_profit"], 2)))
        el = time.time() - t0
        print(f"  {y}  chose " + " ".join(f"{k}={bp[k]}" for k in AXES)
              + f"  ->  OOS ${r['net_profit']:>7,.2f}  dd {dd*100:>4.1f}%   "
                f"(baseline ${rb['net_profit']:>7,.2f})   "
                f"[{n}/{len(windows)}, {el/60:.0f}m]", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "gate1_walkforward.csv", index=False)

    print(f"\n{'='*72}\nGATE 1 -- do the chosen values hold still?\n{'='*72}")
    for k in AXES:
        vc = out[k].value_counts().sort_index()
        share = vc.max() / len(out)
        print(f"  {k:<11} " + "  ".join(f"{v}x{c}" for v, c in vc.items())
              + f"   most common wins {share:.0%} of windows")
    stable = all(out[k].value_counts().max() / len(out) >= 0.5 for k in AXES)

    pos = int((out["oos_net"] > 0).sum())
    print(f"\n  out-of-sample years profitable : {pos}/{len(out)}")
    print(f"  total out-of-sample            : ${out['oos_net'].sum():,.2f}")
    print(f"  same years, frozen baseline    : ${out['baseline_net'].sum():,.2f}")
    print(f"  median year   re-chosen ${out['oos_net'].median():,.2f}   "
          f"baseline ${out['baseline_net'].median():,.2f}")

    print(f"\n  values stay in one region : {'YES' if stable else 'NO'}")
    print(f"  out-of-sample is positive : {'YES' if out['oos_net'].sum() > 0 else 'NO'}")
    print(f"  -> {'PASS' if (stable and out['oos_net'].sum() > 0) else 'FAIL'}")
    print(f"\nsaved -> {RESULTS / 'gate1_walkforward.csv'}")


if __name__ == "__main__":
    main()
