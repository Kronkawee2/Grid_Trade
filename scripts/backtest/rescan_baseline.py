"""
Re-centre the baseline on a region whose neighbours all survive.

The first scan put the current values at the centre of every axis, which
made them the only interior point in the grid and therefore the only one
whose neighbours could be checked at all. That is an artefact of how the
box was drawn, not evidence that they sit anywhere good: 44 of the 243
cells ruined the account, and the current values ranked 92nd of 243.

The axis-by-axis counts pointed somewhere specific. vol_block = 1.00
ruined nothing across all 81 cells containing it, against 9 at 1.15 and
35 at 1.35; atr_fast = 10 ruined 3 against 33 at 50. So the centre moves
there, the ranges are redrawn around it, and vol_block is allowed below
1.00 -- the previous scan had it pinned to the edge, which leaves the
question of what lies outside unanswered.

The winner is NOT the highest-scoring cell. It is the interior point
whose worst neighbour is strongest: a configuration that has to be exact
is one that will fail as soon as live markets differ from the backtest,
and they always do.

EURUSD only. Gold takes no part in choosing anything.

    python scripts/backtest/rescan_baseline.py
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

# Five values per axis, not three, so an interior point has room on both
# sides and the edges are far enough out to be reachable.
AXES = {
    "atr_mult":  [3.8, 4.2, 4.6, 5.0, 5.4],
    "rebound":   [0.35, 0.45, 0.55, 0.65, 0.75],
    "vol_block": [0.80, 0.90, 1.00, 1.10, 1.20],   # now open below 1.00
    "atr_fast":  [5, 10, 15, 20, 30],
    "atr_slow":  [150, 200, 288, 400],
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
    print(f"{SYMBOL} m5, {len(combos)} cells "
          f"({' x '.join(str(len(v)) for v in AXES.values())}), EURUSD only\n",
          flush=True)

    rows, t0 = [], time.time()
    for i, c in enumerate(combos, 1):
        rows.append(measure(df, dict(zip(AXES, c)), years))
        if i % 100 == 0 or i == len(combos):
            el = time.time() - t0
            print(f"  {i:>5}/{len(combos)}  {el/i:.1f}s each  "
                  f"eta {(len(combos)-i)*el/i/60:.0f}m", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "rescan_baseline.csv", index=False)

    print(f"\n{'='*72}")
    print(f"  cells profitable {(out['final'] > CASH).sum()}/{len(out)}   "
          f"ruined {int(out['ruined'].sum())}")
    print("\naxis by axis (median over everything else):")
    for k, vals in AXES.items():
        print(f"  {k:<11}", end="")
        for v in vals:
            s = out[out[k] == v]
            print(f"  {v}: ${s['final'].median():>6,.0f}/{int(s['ruined'].sum()):>2}r", end="")
        print()

    idx = {k: {v: i for i, v in enumerate(vals)} for k, vals in AXES.items()}
    lut = {tuple(r[k] for k in AXES): r for _, r in out.iterrows()}
    cands = []
    for combo in itertools.product(*AXES.values()):
        c = dict(zip(AXES, combo))
        if any(idx[k][c[k]] in (0, len(AXES[k]) - 1) for k in AXES):
            continue
        nb = [lut[combo]]
        for k in AXES:
            for step in (-1, 1):
                alt = dict(c)
                alt[k] = AXES[k][idx[k][c[k]] + step]
                nb.append(lut[tuple(alt[j] for j in AXES)])
        if any(n["ruined"] for n in nb):
            continue
        fin = [float(n["final"]) for n in nb]
        dds = [float(n["max_dd_pct"]) for n in nb]
        cands.append((c, lut[combo], min(fin), float(np.median(fin)), max(dds)))

    print(f"\n{len(cands)} interior points have every neighbour safe")
    if not cands:
        print("  none -- widen the ranges further")
        return

    # ranked by the WORST neighbour, not by the centre's own score
    cands.sort(key=lambda x: -x[2])
    print("\ntop 8 by worst-neighbour strength (this is the selection rule):\n")
    for c, row, worst, med, wdd in cands[:8]:
        print("  " + " ".join(f"{k}={c[k]}" for k in AXES))
        print(f"     centre ${row['final']:>7,.0f}  dd {row['max_dd_pct']:>5.1f}%  "
              f"months {row['months_up_pct']:>4.1f}%  medyr {row['median_year_pct']:>5.2f}%")
        print(f"     worst neighbour ${worst:>7,.0f}   worst neighbour dd {wdd:>5.1f}%   "
              f"median of the 11 ${med:>7,.0f}")

    pick = cands[0][0]
    cur = {k: BASE[k] for k in AXES}
    print(f"\n{'='*72}\nproposed baseline v2 (safest neighbourhood, not highest score)")
    for k in AXES:
        print(f"  {k:<11} {cur[k]}  ->  {pick[k]}")
    b = measure(df, cur, years)
    n = lut[tuple(pick[k] for k in AXES)]
    print(f"\n  current  ${b['final']:>7,.0f}  dd {b['max_dd_pct']:>5.1f}%  "
          f"months {b['months_up_pct']:>4.1f}%  medyr {b['median_year_pct']:>5.2f}%")
    print(f"  proposed ${n['final']:>7,.0f}  dd {n['max_dd_pct']:>5.1f}%  "
          f"months {n['months_up_pct']:>4.1f}%  medyr {n['median_year_pct']:>5.2f}%")
    json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in pick.items()},
              open(ROOT / "baseline_v2_candidate.json", "w"), indent=2)
    print(f"\nsaved -> {RESULTS / 'rescan_baseline.csv'}")
    print(f"saved -> {ROOT / 'baseline_v2_candidate.json'}  (candidate only, not applied)")


if __name__ == "__main__":
    main()
