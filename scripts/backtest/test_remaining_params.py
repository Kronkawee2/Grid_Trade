"""
The last two parameters that have never been tested on this engine.

stop_extra and dd_pause_bars were carried through every scan as fixed
values and never varied. That is the same gap that hid atr_fast, and
atr_fast turned out to matter enormously -- 10 instead of 20 was worth
$877 against $594 with less than half the drawdown. There is no reason to
assume these two are different, and every reason to check.

They also interact with the values around them: stop_extra sets how far
beyond the deepest level the basket stop sits, in grid steps, so widening
the grid moves it too. dd_pause_bars decides how long the system stands
down after the equity stop fires, which is only meaningful relative to
how often that happens.

Judged on gold, the only market that took no part in choosing anything:
no ruin, drawdown at most 20%, at least 55% of months profitable, then
ranked by profit. EURUSD is shown for information, not for the decision.

    python scripts/backtest/test_remaining_params.py
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
START, END, CASH = "2006-01-01", "2024-12-31", 100.0
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
BASE = json.loads((ROOT / "params_100usd.json").read_text())
MAX_LOT = {"EURUSD": 40.0, "XAUUSD": 30.0}

# The baseline value sits in the middle of each list.
AXES = {
    "stop_extra":    [0.5, 1.0, 1.5, 2.5, 4.0],
    "dd_pause_bars": [50, 150, 350, 800, 2000],
}
HELD = {k: BASE[k] for k in
        ("atr_mult", "rebound", "vol_block", "vol_floor", "atr_fast", "atr_slow",
         "max_open", "equity_stop", "side", "er_window", "er_block",
         "dd_pause", "max_hold_days")}

# Gate B, as agreed: gold decides, and drawdown is the hard limit.
DD_LIMIT, MONTHS_MIN = 20.0, 55.0


def money(spec):
    s = SPECS[spec]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=MAX_LOT[spec])


def measure(df, spec, p):
    res = run_adaptive(df, AdaptiveConfig(**HELD, **p, **money(spec)))
    eq = CASH + res["equity"]
    peak = eq.cummax()
    monthly = res["equity"].resample("ME").last().dropna().diff().dropna()
    ye = eq.resample("YE").last()
    ypct = (ye / ye.shift(1).fillna(CASH) - 1) * 100
    return dict(final=round(float(eq.iloc[-1]), 2),
                dd=round(float(((peak - eq) / peak).max()) * 100, 2),
                months=round(float((monthly > 0).mean()) * 100, 1),
                med_year=round(float(ypct.median()), 2),
                years_up=int((ypct > 0).sum()),
                trades=res["n_trades"],
                ruined=bool(eq.min() <= 0))


def main():
    RESULTS.mkdir(exist_ok=True)
    data = {s: load_bars(s, "m5", start=START, end=END)
            for s in ("EURUSD.DUKA", "XAUUSD.DUKA")}
    combos = list(itertools.product(*AXES.values()))
    print(f"{len(combos)} combinations, both markets. gold decides.\n")
    print(f"{'stop_extra':>10} {'pause':>7} | {'EURUSD':^22} | {'GOLD (judge)':^22} | verdict")
    print(f"{'':>10} {'':>7} | {'final':>8} {'dd':>6} {'mo+':>6} | "
          f"{'final':>8} {'dd':>6} {'mo+':>6} |")
    print("-" * 84)

    rows = []
    t0 = time.time()
    for se, dp in combos:
        p = dict(stop_extra=se, dd_pause_bars=dp)
        e = measure(data["EURUSD.DUKA"], "EURUSD", p)
        g = measure(data["XAUUSD.DUKA"], "XAUUSD", p)
        ok = (not g["ruined"]) and g["dd"] <= DD_LIMIT and g["months"] >= MONTHS_MIN
        mark = "  <- baseline" if (se == BASE["stop_extra"]
                                   and dp == BASE["dd_pause_bars"]) else ""
        rows.append(dict(stop_extra=se, dd_pause_bars=dp, passes=ok,
                         **{f"eur_{k}": v for k, v in e.items()},
                         **{f"gold_{k}": v for k, v in g.items()}))
        print(f"{se:>10} {dp:>7} | {e['final']:>8,.0f} {e['dd']:>5.1f}% {e['months']:>5.1f}% | "
              f"{g['final']:>8,.0f} {g['dd']:>5.1f}% {g['months']:>5.1f}% | "
              f"{'PASS' if ok else 'fail'}{mark}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "test_remaining_params.csv", index=False)
    ok = out[out["passes"]]
    print(f"\n{len(ok)} of {len(out)} pass the gate  "
          f"({time.time()-t0:.0f}s)")

    b = out[(out.stop_extra == BASE["stop_extra"])
            & (out.dd_pause_bars == BASE["dd_pause_bars"])].iloc[0]
    print(f"\nbaseline (stop_extra {BASE['stop_extra']}, "
          f"dd_pause_bars {BASE['dd_pause_bars']}):")
    print(f"  gold ${b.gold_final:,.0f}  dd {b.gold_dd}%  months {b.gold_months}%  "
          f"{'PASS' if b.passes else 'FAIL'}")
    if len(ok):
        rank = int((ok["gold_final"] > b.gold_final).sum()) + 1
        print(f"  ranks {rank} of {len(ok)} among those that pass")

    print("\neach axis, median over the other:")
    for k, vals in AXES.items():
        print(f"  {k}:")
        for v in vals:
            s = out[out[k] == v]
            print(f"    {v:>6}: gold ${s.gold_final.median():>7,.0f}  "
                  f"dd {s.gold_dd.median():>5.1f}%  months {s.gold_months.median():>5.1f}%  "
                  f"passing {int(s.passes.sum())}/{len(s)}")

    if len(ok):
        top = ok.nlargest(5, "gold_final")
        print("\ntop 5 by gold profit, among those that pass:")
        print(top[["stop_extra", "dd_pause_bars", "gold_final", "gold_dd",
                   "gold_months", "eur_final", "eur_dd"]].to_string(index=False))
    print(f"\nsaved -> {RESULTS / 'test_remaining_params.csv'}")


if __name__ == "__main__":
    main()
