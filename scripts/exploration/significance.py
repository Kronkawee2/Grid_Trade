"""
Three statistical gates on the baseline, plus one the baseline needs.

A backtest that shows a profit has shown one thing: that this strategy,
on this history, made money. It has not shown the money came from an
edge. These tests ask that question three different ways, and a
configuration has to survive all of them before the word "edge" is
justified.

    1. Bootstrap CI          is the average trade profitable, or is the
                             total carried by a handful of outliers?
    2. Monte Carlo           does it beat entering at random, holding for
                             the same length of time, the same number of
                             times?
    3. Fixed-config folds    does one locked configuration hold up across
                             every stretch of history, or only some?

The fourth is a block bootstrap, added because this strategy's profits
are known to arrive in bursts -- four years out of nineteen carry EURUSD,
and gold's last four years carry 94% of its total. Resampling individual
trades scatters those bursts and reports a confidence the record does not
support. Resampling contiguous months keeps them intact.

Tuning happens on EURUSD only. Gold is never used to choose anything, so
its numbers here are a verification and not a second opinion.

    python scripts/backtest/significance.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
START, END, CASH = "2006-01-01", "2024-12-31", 100.0
N_BOOT, N_MC = 5000, 1000
BLOCK_MONTHS = 3
FOLD_YEARS = 2

PARAMS = json.loads((ROOT / "params_100usd.json").read_text())
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
GRID = {k: PARAMS[k] for k in
        ("atr_mult", "max_open", "stop_extra", "rebound", "vol_block", "vol_floor",
         "atr_fast", "atr_slow", "side", "er_window", "er_block",
         "dd_pause", "dd_pause_bars", "max_hold_days")}
GRID["equity_stop"] = PARAMS["equity_stop"]
MAX_LOT = {"EURUSD": 40.0, "XAUUSD": 30.0}
MARKETS = [("EURUSD", "EURUSD.DUKA", "tuned on this"),
           ("XAUUSD", "XAUUSD.DUKA", "never used to tune")]


def money(spec):
    s = SPECS[spec]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=MAX_LOT[spec])


def bootstrap_trades(net, rng):
    """Resample individual trades. Assumes they are independent -- they are not."""
    draws = rng.choice(net, size=(N_BOOT, len(net)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 50, 97.5]), float((draws > 0).mean())


def bootstrap_blocks(tr, rng):
    """
    Resample contiguous three-month blocks instead of single trades, so a
    burst of profit stays together with the months that produced it.
    """
    key = pd.to_datetime(tr["closed_at"]).dt.to_period("M")
    months = sorted(key.unique())
    blocks = [np.concatenate([tr.loc[key == m, "net_cash"].to_numpy()
                              for m in months[i:i + BLOCK_MONTHS]])
              for i in range(0, len(months) - BLOCK_MONTHS + 1)]
    blocks = [b for b in blocks if len(b)]
    need = int(np.ceil(len(months) / BLOCK_MONTHS))
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(blocks), size=need)
        out[i] = np.concatenate([blocks[j] for j in pick]).sum()
    return np.percentile(out, [2.5, 50, 97.5]), float((out > 0).mean())


def monte_carlo(df, tr, spec, rng):
    """
    Random entries with the strategy's own trade count, direction mix and
    holding times, priced through the same costs. If the strategy cannot
    beat this, its entries carry no information.
    """
    kw = money(spec)
    pip, pv, spread = kw["pip"], kw["pip_value_per_lot"] * kw["lot"], kw["spread_pips"]
    swap = {"buy": kw["swap_long_pips"], "sell": kw["swap_short_pips"]}
    live = df.index[df.high > df.low]
    pos = pd.Series(np.arange(len(df)), index=df.index)
    close = df["close"].to_numpy()

    held = (pd.to_datetime(tr["closed_at"]) - pd.to_datetime(tr["opened_at"]))
    bars_held = np.maximum((held.dt.total_seconds() / 300).round().astype(int), 1).to_numpy()
    sides = tr["side"].to_numpy()
    n = len(tr)

    totals = np.empty(N_MC)
    starts_pool = pos.reindex(live).to_numpy()
    for i in range(N_MC):
        s = rng.choice(starts_pool, size=n, replace=True)
        e = np.minimum(s + bars_held, len(df) - 1)
        d = np.where(sides == "buy", 1.0, -1.0)
        gross = (close[e] - close[s]) / pip * d
        nights = bars_held / 288.0
        sw = np.where(sides == "buy", swap["buy"], swap["sell"]) * nights
        totals[i] = ((gross - spread + sw) * pv).sum()
    return totals


def folds(df, spec, res):
    """One locked configuration, every two-year stretch, scored separately."""
    out = []
    y0, y1 = df.index[0].year, df.index[-1].year
    for a in range(y0, y1 + 1, FOLD_YEARS):
        b = min(a + FOLD_YEARS - 1, y1)
        d = df.loc[f"{a}":f"{b}"]
        if len(d) < 20000:
            continue
        r = run_adaptive(d, AdaptiveConfig(**GRID, **money(spec)))
        eq = CASH + r["equity"]
        peak = eq.cummax()
        out.append(dict(fold=f"{a}-{b}", net=round(r["net_profit"], 2),
                        dd_pct=round(float(((peak - eq) / peak).max()) * 100, 1),
                        trades=r["n_trades"]))
    return pd.DataFrame(out)


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    rows = []

    for spec, sym, note in MARKETS:
        df = load_bars(sym, "m5", start=START, end=END)
        res = run_adaptive(df, AdaptiveConfig(**GRID, **money(spec)))
        tr = res["trades"]
        net = tr["net_cash"].to_numpy()
        total = float(net.sum())

        print(f"\n{'='*74}\n{spec}  ({note})   {len(tr):,} trades   "
              f"net ${total:,.2f}\n{'='*74}")

        (lo, mid, hi), pos = bootstrap_trades(net, rng)
        pass1 = lo > 0
        print(f"\n1. Bootstrap CI on single trades ({N_BOOT:,} resamples)")
        print(f"   mean per trade   95% CI  ${lo:+.4f} .. ${hi:+.4f}   median ${mid:+.4f}")
        print(f"   positive in {pos:.1%} of resamples")
        print(f"   -> {'PASS' if pass1 else 'FAIL'} (needs the lower bound above zero)")

        (blo, bmid, bhi), bpos = bootstrap_blocks(tr, rng)
        pass1b = blo > 0
        print(f"\n1b. Block bootstrap, {BLOCK_MONTHS}-month blocks ({N_BOOT:,} resamples)")
        print(f"   total profit     95% CI  ${blo:+,.0f} .. ${bhi:+,.0f}   median ${bmid:+,.0f}")
        print(f"   positive in {bpos:.1%} of resamples")
        print(f"   -> {'PASS' if pass1b else 'FAIL'}")

        mc = monte_carlo(df, tr, spec, rng)
        beat = float((total > mc).mean())
        pass2 = beat >= 0.95
        print(f"\n2. Monte Carlo against random entry ({N_MC:,} runs)")
        print(f"   random: median ${np.median(mc):,.0f}   "
              f"95th pct ${np.percentile(mc, 95):,.0f}")
        print(f"   strategy ${total:,.0f} beats {beat:.1%} of them")
        print(f"   -> {'PASS' if pass2 else 'FAIL'} (needs 95%)")

        fd = folds(df, spec, res)
        good = int((fd["net"] > 0).sum())
        pass3 = good >= len(fd) - 1
        print(f"\n3. Fixed-config walk-forward, {FOLD_YEARS}-year folds")
        for _, f in fd.iterrows():
            print(f"   {f.fold}   net ${f.net:>8,.2f}   maxDD {f.dd_pct:>5.1f}%   "
                  f"{f.trades:>5,} trades")
        print(f"   profitable folds {good}/{len(fd)}")
        print(f"   -> {'PASS' if pass3 else 'FAIL'} (allows one losing fold)")

        verdict = all([pass1, pass1b, pass2, pass3])
        print(f"\n   OVERALL: {'PASSES ALL GATES' if verdict else 'DOES NOT PASS'}")
        rows.append(dict(market=spec, trades=len(tr), net=round(total, 2),
                         boot_lo=round(lo, 4), boot_pass=pass1,
                         block_lo=round(blo, 1), block_pass=pass1b,
                         mc_beat_pct=round(beat * 100, 1), mc_pass=pass2,
                         folds_positive=good, folds_total=len(fd), fold_pass=pass3,
                         overall=verdict))
        fd.to_csv(RESULTS / f"folds_{spec}.csv", index=False)

    pd.DataFrame(rows).to_csv(RESULTS / "significance.csv", index=False)
    print(f"\nsaved -> {RESULTS / 'significance.csv'}")


if __name__ == "__main__":
    main()
