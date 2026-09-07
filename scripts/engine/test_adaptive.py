"""
Known-answer tests for the adaptive engine.

A backtest cannot be checked by looking at its profit, because any number
it prints looks plausible. The only way to know the engine is right is to
feed it paths whose correct answer can be worked out by hand first, and
then compare.

Four defects have already been found in this file, and every one of them
was noticed by accident -- a figure that looked wrong, or a question that
happened to be asked. A defect that does not make the numbers look
strange would still be here. Each of those four now has a test below, so
they cannot come back silently.

    python scripts/backtest/test_adaptive.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive import (AdaptiveConfig, run_adaptive, true_range,  # noqa: E402
                      volatility, efficiency_ratio)

PIP = 0.0001
FAILURES = []


def bars(prices, start="2020-01-06", freq="5min"):
    """Each bar's range is exactly the move it makes -- no invented wicks."""
    prices = np.asarray(prices, dtype=float)
    idx = pd.date_range(start, periods=len(prices), freq=freq)
    prev = np.concatenate([[prices[0]], prices[:-1]])
    return pd.DataFrame({"open": prev,
                         "high": np.maximum(prev, prices),
                         "low": np.minimum(prev, prices),
                         "close": prices,
                         "volume": 100}, index=idx)


def sawtooth(centre, amp_pips, n, freq="5min", start="2020-01-06"):
    """Alternate up and down by a fixed amount: constant, known volatility."""
    a = amp_pips * PIP
    return bars([centre + (a if i % 2 else -a) for i in range(n)],
                start=start, freq=freq)


def cfg(**kw):
    base = dict(atr_mult=5.0, max_open=3, stop_extra=1.5, rebound=0.0,
                vol_block=99.0, vol_floor=0.0, atr_fast=10, atr_slow=200,
                side="both", er_window=0, er_block=1.0, dd_pause=None,
                dd_pause_bars=0, max_hold_days=None, equity_stop=None,
                pip=PIP, pip_value_per_lot=10.0, lot=0.01, spread_pips=0.0,
                swap_long_pips=0.0, swap_short_pips=0.0, start_cash=100.0)
    base.update(kw)
    return AdaptiveConfig(**base)


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# 1. Spacing is exactly atr_mult x ATR. If this is wrong nothing else means
#    anything, because every other distance is a multiple of it.
# --------------------------------------------------------------------------
def test_spacing_follows_atr():
    print("\n1. spacing = atr_mult x ATR")
    # a +/-amp sawtooth travels 2*amp between consecutive bars, so the
    # true range -- and therefore the ATR -- is 2*amp, not amp.
    for amp, mult in ((4.0, 5.0), (4.0, 3.0), (10.0, 5.0)):
        df = sawtooth(1.1000, amp, 600)
        r = run_adaptive(df, cfg(atr_mult=mult))
        spacing = r["boxes"]["spacing"].iloc[-1] / PIP
        check(f"amp {amp}p x{mult}", round(spacing, 3), round(2 * amp * mult, 3), 0.05)


# --------------------------------------------------------------------------
# 2. Double the volatility, double the grid -- the property the whole
#    "adaptive" claim rests on.
# --------------------------------------------------------------------------
def test_spacing_scales():
    print("\n2. twice the volatility gives twice the spacing")
    a = run_adaptive(sawtooth(1.1000, 4.0, 600), cfg())
    b = run_adaptive(sawtooth(1.1000, 8.0, 600), cfg())
    sa = a["boxes"]["spacing"].iloc[-1] / PIP
    sb = b["boxes"]["spacing"].iloc[-1] / PIP
    check("ratio", round(sb / sa, 2), 2.0, 0.05)


# --------------------------------------------------------------------------
# 3. Never more than max_open positions, whatever the price does.
# --------------------------------------------------------------------------
def test_position_cap():
    print("\n3. the cap on open positions holds")
    path = list(np.linspace(1.1000, 1.0000, 400))     # straight down through every level
    df = bars(path)
    for cap in (1, 2, 3):
        r = run_adaptive(df, cfg(max_open=cap, stop_extra=99.0))
        check(f"cap {cap}", r["max_concurrent"], cap)


# --------------------------------------------------------------------------
# 4. REGRESSION -- a position may not close on the bar that opened it.
#    This was the first defect: 86-92% of equity-stop exits landed on their
#    own opening bar and booked a POSITIVE average gross result, which is a
#    forced loss that made money.
# --------------------------------------------------------------------------
def test_no_same_bar_exit():
    print("\n4. no position closes on its opening bar (regression)")
    rng = np.random.default_rng(3)
    walk = 1.1000 + np.cumsum(rng.normal(0, 6 * PIP, 3000))
    df = bars(walk)
    for lbl, kw in (("no equity stop", {}), ("equity stop $2", dict(equity_stop=2.0)),
                    ("tight stop", dict(stop_extra=0.2))):
        r = run_adaptive(df, cfg(**kw))
        tr = r["trades"]
        same = int((tr["opened_at"] == tr["closed_at"]).sum()) if len(tr) else 0
        check(f"{lbl} ({len(tr)} trades)", same, 0)


# --------------------------------------------------------------------------
# 5. REGRESSION -- true range must not reach across a hole in the data.
#    466 missing days on gold produced true ranges 10x to 100x normal,
#    which then set the grid spacing for the next 200 bars.
# --------------------------------------------------------------------------
def test_true_range_across_gap():
    print("\n5. true range ignores a gap in the data (regression)")
    step = 4.0 * PIP
    a = bars([1.1000 + i * step for i in range(100)])
    b = bars([1.2000 + i * step for i in range(100)], start="2020-06-01")
    # `bars` gives its first row open == close, so the bar after the join
    # would have no range of its own and nothing to check. Give it one.
    b.iloc[0, b.columns.get_loc("open")] = 1.2000 - step
    b.iloc[0, b.columns.get_loc("low")] = 1.2000 - step
    df = pd.concat([a, b])
    tr = true_range(df) / PIP
    i = len(a)                                             # first bar after the hole
    # without the guard this bar reads ~1000 pips: the jump between the
    # two stretches booked as a single bar's range
    check("bar after a 5-month gap", round(float(tr[i]), 1), 4.0, 0.6)
    check("nothing above 2x normal", round(float(np.nanmax(tr)), 1) <= 8.0, True)
    check("no NaN anywhere", int(np.isnan(tr).sum()), 0)


# --------------------------------------------------------------------------
# 6. REGRESSION -- the broker's maximum lot must be respected. Without it a
#    compounding run on gold reached 207 lots against a 30-lot ceiling and
#    reported turning $100 into two million.
# --------------------------------------------------------------------------
def test_lot_ceiling():
    print("\n6. lot never exceeds the broker maximum (regression)")
    rng = np.random.default_rng(7)
    walk = 1.1000 + np.cumsum(rng.normal(0.4 * PIP, 6 * PIP, 4000))
    r = run_adaptive(bars(walk), cfg(compound=True, max_lot=0.05,
                                     equity_per_lot_step=1.0))
    check("max lot traded", float(r["trades"]["lot"].max()), 0.05)


# --------------------------------------------------------------------------
# 7. REGRESSION -- the equity stop must buy the same price distance at any
#    lot size. Held fixed while the lot compounded it shrank to a hundredth
#    of a pip, which is inside the spread and therefore meaningless.
# --------------------------------------------------------------------------
def test_equity_stop_scales_with_lot():
    print("\n7. equity stop keeps its distance as the lot grows (regression)")
    rng = np.random.default_rng(11)
    walk = 1.1000 + np.cumsum(rng.normal(0, 6 * PIP, 3000))
    df = bars(walk)
    base = run_adaptive(df, cfg(equity_stop=3.0, lot=0.01))
    # ten times the lot, ten times the cash cap -> identical trade count
    ten = run_adaptive(df, cfg(equity_stop=30.0, lot=0.10))
    check("same trades at 10x lot and 10x cap", ten["n_trades"], base["n_trades"])
    check("ten times the money", round(ten["net_profit"] / base["net_profit"], 3)
          if base["net_profit"] else 0.0, 10.0, 0.01)


# --------------------------------------------------------------------------
# 8. Costs land where they should: spread once per trade, swap per night.
# --------------------------------------------------------------------------
def test_costs():
    print("\n8. spread and swap")
    # a two-point oscillation cannot fill a grid five ATRs wide -- the
    # path has to actually travel
    rng = np.random.default_rng(5)
    df = bars(1.1000 + np.cumsum(rng.normal(0, 6 * PIP, 3000)))
    free = run_adaptive(df, cfg(spread_pips=0.0))
    paid = run_adaptive(df, cfg(spread_pips=2.0))
    n = free["n_trades"]
    lost = (free["net_profit"] - paid["net_profit"]) / (n * 10.0 * 0.01) if n else 0
    check(f"2p spread costs 2p per trade ({n} trades)", round(lost, 3), 2.0, 0.02)


# --------------------------------------------------------------------------
# 9. A market that only falls must lose money on a two-sided grid. If it
#    does not, the engine is inventing money somewhere.
# --------------------------------------------------------------------------
def test_one_way_market_loses():
    print("\n9. a one-way market loses")
    down = run_adaptive(bars(list(np.linspace(1.1000, 1.0000, 800))), cfg())
    check("straight down", down["net_profit"] < 0, True)
    up = run_adaptive(bars(list(np.linspace(1.1000, 1.2000, 800))), cfg())
    check("straight up", up["net_profit"] < 0, True)


# --------------------------------------------------------------------------
# 10. The volatility gate blocks what it says it blocks.
# --------------------------------------------------------------------------
def test_vol_gate():
    print("\n10. the volatility gate")
    calm = sawtooth(1.1000, 3.0, 400)
    wild = sawtooth(1.1000, 30.0, 200, start="2020-01-08")
    df = pd.concat([calm, wild])
    r_open = run_adaptive(df, cfg(vol_block=99.0))
    r_gate = run_adaptive(df, cfg(vol_block=1.2))
    check("gate reduces trading", r_gate["n_trades"] <= r_open["n_trades"], True)
    check("gate records blocked bars", r_gate["n_blocked_bars"] > 0, True)


# --------------------------------------------------------------------------
# 11. Market-closed bars (high == low) take no part in anything.
# --------------------------------------------------------------------------
def test_closed_market_ignored():
    print("\n11. bars with no range are ignored")
    # a random walk, not a sawtooth: the price has to travel further than
    # the grid is wide or nothing can ever fill
    rng = np.random.default_rng(13)
    live = bars(1.1000 + np.cumsum(rng.normal(0, 6 * PIP, 2000)))
    # the shut stretch has to come AFTER the live one, on timestamps of its
    # own. Overlapping them gives duplicate index entries and the question
    # "was this trade opened while the market was shut" stops having an
    # answer.
    shut = bars([float(live["close"].iloc[-1])] * 100,
                start=live.index[-1] + pd.Timedelta("5D"))
    for c in ("open", "high", "low", "close"):
        shut[c] = float(live["close"].iloc[-1])   # a flat, closed market
    df = pd.concat([live, shut])
    check("no duplicate timestamps", int(df.index.duplicated().sum()), 0)
    r = run_adaptive(df, cfg())
    check("the run traded at all", r["n_trades"] > 0, True)
    if not r["n_trades"]:
        return
    opened = pd.to_datetime(r["trades"]["opened_at"])
    check("no trade opened in the shut stretch",
          int(((opened >= shut.index[0]) & (opened <= shut.index[-1])).sum()), 0)


# --------------------------------------------------------------------------
# 12. Efficiency ratio: 1 for a straight line, near 0 for a sawtooth.
# --------------------------------------------------------------------------
def test_efficiency_ratio():
    print("\n12. efficiency ratio")
    line = bars(list(np.linspace(1.1000, 1.1100, 200)))
    saw = sawtooth(1.1000, 4.0, 200)
    check("straight line", round(float(efficiency_ratio(line, 60)[-1]), 2), 1.0, 0.02)
    check("sawtooth", round(float(efficiency_ratio(saw, 60)[-1]), 2) < 0.1, True)


if __name__ == "__main__":
    test_spacing_follows_atr()
    test_spacing_scales()
    test_position_cap()
    test_no_same_bar_exit()
    test_true_range_across_gap()
    test_lot_ceiling()
    test_equity_stop_scales_with_lot()
    test_costs()
    test_one_way_market_loses()
    test_vol_gate()
    test_closed_market_ignored()
    test_efficiency_ratio()
    print("\n" + ("ALL PASS" if not FAILURES else f"FAILED: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)
