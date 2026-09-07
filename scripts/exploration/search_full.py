"""
Gate 2 and 3: search every parameter at once, on the engine actually in use.

The configuration in params_100usd.json came from a 45-cell scan that
varied three parameters and froze six. Two of the frozen six -- atr_fast
and atr_slow -- had never been tested at all, and one of them turned out
to matter enormously: atr_fast=10 returns $877 at a 16% drawdown where
the current 20 returns $594 at 36%. Worse, that scan ran before the
equity stop existed, and the equity stop is the single component without
which both markets are ruined. So the plateau it found belongs to a
different system than the one being run.

This closes both gaps by searching all nine parameters together, on the
current engine, with nothing held back.

EURUSD only. Gold is never used to choose anything -- it is the held-out
market, and a result there is evidence precisely because it took no part
in the fitting.

Every metric is stored for every trial rather than collapsed into one
score, so the choice of what "best" means can be made after seeing the
results instead of before.

    python scripts/backtest/search_full.py --trials 600
"""

import argparse
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
START, END, CASH = "2006-01-01", "2024-12-31", 100.0
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())

# Ranges are wide on purpose. A winner that lands on an edge means the box
# was drawn in the wrong place, and that is worth finding out now rather
# than after another set of conclusions has been built on top.
SPACE = {
    "atr_mult":      ("float",  2.0, 12.0),
    "rebound":       ("float",  0.0, 0.8),
    "vol_block":     ("float",  0.8, 3.0),
    "stop_extra":    ("float",  0.5, 5.0),
    "equity_stop":   ("float",  8.0, 50.0),
    # 288 bars is one trading day on m5. The current 200 is seventeen
    # hours, which is why the gate ended up reading the clock.
    "atr_slow":      ("choice", [100, 200, 288, 576, 1440, 2880]),
    "atr_fast":      ("choice", [5, 10, 20, 50, 100]),
    "max_open":      ("choice", [2, 3]),          # the user's cap
    "dd_pause_bars": ("choice", [100, 350, 800, 2000]),
}
FIXED = dict(side="both", vol_floor=0.0, er_window=0, er_block=1.0,
             dd_pause=None, max_hold_days=None, block_hours=())

# Roughly one trade a week. Below this a configuration has not been tested
# by the history, it has merely avoided it -- and the search will happily
# pick "barely trade" because not trading cannot lose.
MIN_TRADES = 1000


def money():
    s = SPECS[SPEC]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=40.0)


def sample(rng):
    out = {}
    for k, spec in SPACE.items():
        if spec[0] == "float":
            out[k] = round(float(rng.uniform(spec[1], spec[2])), 3)
        else:
            out[k] = spec[1][rng.integers(len(spec[1]))]
    if out["atr_slow"] < out["atr_fast"] * 3:      # a baseline must be slower
        out["atr_slow"] = out["atr_fast"] * 3
    return out


def measure(df, params, years):
    res = run_adaptive(df, AdaptiveConfig(**params, **FIXED, **money()))
    eq = CASH + res["equity"]
    peak = eq.cummax()
    dd = float(((peak - eq) / peak).max())
    monthly = res["equity"].resample("ME").last().dropna().diff().dropna()
    yearly = eq.resample("YE").last()
    ypct = (yearly / yearly.shift(1).fillna(CASH) - 1) * 100
    flat = int((eq.resample("1D").last().dropna()
                .pipe(lambda s: (s < s.cummax() - 1e-9)).sum()))
    ok = res["n_trades"] >= MIN_TRADES and dd > 0 and eq.min() > 0
    return {
        **params,
        "final": round(float(eq.iloc[-1]), 2),
        "max_dd_pct": round(dd * 100, 2),
        "net_over_dd": round(res["net_profit"] / (dd * CASH), 3) if dd else np.nan,
        "months_up_pct": round(float((monthly > 0).mean()) * 100, 1),
        "median_year_pct": round(float(ypct.median()), 2),
        "worst_year_pct": round(float(ypct.min()), 2),
        "years_up": int((ypct > 0).sum()),
        "days_flat_pct": round(flat / len(eq.resample("1D").last().dropna()) * 100, 1),
        "trades_per_year": round(res["n_trades"] / years, 0),
        "win_pct": round(res["win_rate"], 1),
        "valid": bool(ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    df = load_bars(SYMBOL, "m5", start=START, end=END)
    years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"{SYMBOL} m5: {len(df):,} bars, {years:.1f} years")
    print(f"searching {len(SPACE)} parameters at once, {args.trials} trials, "
          f"EURUSD only\n", flush=True)

    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for n in range(1, args.trials + 1):
        rows.append(measure(df, sample(rng), years))
        if n % 25 == 0 or n == args.trials:
            d = pd.DataFrame(rows)
            v = d[d["valid"]]
            el = time.time() - t0
            msg = (f"  {n:>4}/{args.trials} | {el/n:.1f}s each | "
                   f"eta {(args.trials-n)*el/n/60:.0f}m | usable {len(v)}")
            if len(v):
                b = v.loc[v["net_over_dd"].idxmax()]
                msg += (f" | best ${b['final']:,.0f} dd {b['max_dd_pct']:.0f}% "
                        f"months {b['months_up_pct']:.0f}%")
            print(msg, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "search_full_eurusd.csv", index=False)
    v = out[out["valid"]]
    print(f"\n{len(v)} of {len(out)} trials usable "
          f"(needed {MIN_TRADES:,}+ trades and no ruin)\n")

    for crit, label in (("net_over_dd", "return per unit of drawdown"),
                        ("months_up_pct", "share of profitable months"),
                        ("median_year_pct", "median year"),
                        ("final", "final balance")):
        top = v.nlargest(5, crit)
        print(f"top 5 by {label}:")
        print(top[["atr_mult","rebound","vol_block","atr_fast","atr_slow",
                   "stop_extra","max_open","equity_stop","final","max_dd_pct",
                   "months_up_pct","median_year_pct","trades_per_year"]]
              .to_string(index=False))
        print()

    cur = dict(atr_mult=4.2, rebound=0.45, vol_block=1.15, atr_fast=20,
               atr_slow=200, stop_extra=1.5, max_open=3, equity_stop=22.0,
               dd_pause_bars=350)
    c = measure(df, cur, years)
    print("the current baseline, measured the same way:")
    print(f"  ${c['final']:,.0f}  dd {c['max_dd_pct']}%  "
          f"net/dd {c['net_over_dd']}  months {c['months_up_pct']}%  "
          f"median year {c['median_year_pct']}%")
    for crit in ("net_over_dd", "months_up_pct", "median_year_pct", "final"):
        rank = int((v[crit] > c[crit]).sum()) + 1
        print(f"  ranks {rank} of {len(v)} on {crit}")
    print(f"\nsaved -> {RESULTS / 'search_full_eurusd.csv'}")


if __name__ == "__main__":
    main()
