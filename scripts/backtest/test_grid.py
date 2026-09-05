"""
Known-answer tests for the grid engine.

A backtest cannot be checked by looking at its profit, because any number
it prints looks plausible. The only way to know the engine is right is to
feed it paths whose correct answer can be worked out by hand first, and
then compare.

    python scripts/backtest/test_grid.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import GridConfig, run_grid  # noqa: E402

PIP = 0.0001
FAILURES = []


def bars(prices, start="2020-01-01", freq="15min"):
    """Build a frame where each bar's range is exactly the move it makes."""
    prices = np.asarray(prices, dtype=float)
    idx = pd.date_range(start, periods=len(prices), freq=freq)
    prev = np.concatenate([[prices[0]], prices[:-1]])
    return pd.DataFrame({
        "open": prev,
        "high": np.maximum(prev, prices),
        "low": np.minimum(prev, prices),
        "close": prices,
        "spread_avg": 0.0,          # costs off, so pips are pure geometry
    }, index=idx)


def cfg(**kw):
    base = dict(spacing_pips=10.0, n_levels=3, side="buy", pip=PIP,
                pip_value=1.0, swap_long_pips=0.0, swap_short_pips=0.0,
                spread_override_pips=0.0)
    base.update(kw)
    return GridConfig(**base)


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# 1. One level down and back: exactly one round trip, +10 pips.
# --------------------------------------------------------------------------
def test_single_round_trip():
    print("\n1. one level down, one level back")
    df = bars([1.0000, 0.9990, 1.0000])
    r = run_grid(df, cfg())
    check("trades", r["n_trades"], 1)
    check("net pips", r["trades"]["net_pips"].sum(), 10.0, 1e-6)


# --------------------------------------------------------------------------
# 2. A sawtooth that dips one level and recovers, repeated ten times.
#    Ten identical round trips, no more and no less.
# --------------------------------------------------------------------------
def test_sawtooth_repeats():
    print("\n2. same dip repeated ten times")
    path = [1.0000]
    for _ in range(10):
        path += [0.9990, 1.0000]
    r = run_grid(bars(path), cfg())
    check("trades", r["n_trades"], 10)
    check("net pips", r["trades"]["net_pips"].sum(), 100.0, 1e-6)


# --------------------------------------------------------------------------
# 3. THE ANCHOR TEST.
#    A round trip completes, the market then moves 100 pips away, and
#    sawtooths there. Because the grid is flat when the market moves, the
#    anchor should follow and keep trading in the new range.
# --------------------------------------------------------------------------
def test_reanchor_while_flat():
    print("\n3. market moves while flat, grid follows")
    path = [1.0000, 0.9990, 1.0000]      # one clean round trip at the anchor
    path += [1.0100]                      # market steps up 100 pips, grid flat
    for _ in range(10):                   # and sawtooths there
        path += [1.0090, 1.0100]
    r = run_grid(bars(path), cfg(n_levels=3, stop_pips=1000.0))
    print(f"     resets={r['n_resets']}  trades={r['n_trades']}  "
          f"open_at_end={r['open_at_end']}")
    check("resets happened", r["n_resets"] > 0, True)
    check("trades in new range", r["n_trades"], 11)


# --------------------------------------------------------------------------
# 3b. The other half of the same rule: while a position is open the grid
#     must NOT chase the price. Deliberate, and the source of grid risk.
# --------------------------------------------------------------------------
def test_frozen_while_open():
    print("\n3b. grid freezes while holding")
    path = [1.0000, 0.9990]              # one level filled, stays open
    path += list(np.linspace(0.9985, 0.9500, 60))   # market walks away
    r = run_grid(bars(path), cfg(n_levels=3, stop_pips=1000.0))
    print(f"     resets={r['n_resets']}  trades={r['n_trades']}  "
          f"open_at_end={r['open_at_end']}")
    # Only the three levels below the original anchor may ever fill.
    check("no more than n_levels open", r["open_at_end"], 3)
    check("no resets while holding", r["n_resets"], 0)


# --------------------------------------------------------------------------
# 4. Stop fires at the right price and flattens everything.
# --------------------------------------------------------------------------
def test_stop():
    print("\n4. stop")
    # spacing 10, 3 levels, stop 60 pips below anchor
    df = bars([1.0000, 0.9990, 0.9980, 0.9970, 0.9930])
    r = run_grid(df, cfg(stop_pips=60.0))
    stops = r["trades"][r["trades"]["reason"] == "stop"]
    check("all three positions stopped", len(stops), 3)
    check("stop exit price", float(stops["exit"].iloc[0]), 0.9940, 1e-9)
    check("nothing left open", r["open_at_end"], 0)


# --------------------------------------------------------------------------
# 5. Costs land where they should: spread once per trade, swap per night.
# --------------------------------------------------------------------------
def test_costs():
    print("\n5. spread and swap")
    df = bars([1.0000, 0.9990, 1.0000])
    r = run_grid(df, cfg(spread_override_pips=2.0))
    check("net after 2p spread", r["trades"]["net_pips"].sum(), 8.0, 1e-6)

    # same trip, but held five days
    daily = bars([1.0000, 0.9990, 1.0000], freq="D")
    daily.index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-07"])
    r = run_grid(daily, cfg(swap_long_pips=-1.0))
    check("swap charged for 5 nights", r["trades"]["swap_pips"].iloc[0], -5.0, 1e-6)


# --------------------------------------------------------------------------
# 6. A market that only falls must lose. If it does not, the engine is
#    inventing money somewhere.
# --------------------------------------------------------------------------
def test_pure_downtrend_loses():
    print("\n6. straight line down (buy grid must lose)")
    path = list(np.linspace(1.0000, 0.9000, 200))
    r = run_grid(bars(path), cfg(n_levels=10, stop_pips=1000.0))
    equity_end = float(r["equity"].iloc[-1])
    print(f"     resets={r['n_resets']}  trades={r['n_trades']}  "
          f"final equity ${equity_end:,.2f}")
    check("loses money", equity_end < 0, True)


if __name__ == "__main__":
    test_single_round_trip()
    test_sawtooth_repeats()
    test_reanchor_while_flat()
    test_frozen_while_open()
    test_stop()
    test_costs()
    test_pure_downtrend_loses()
    print("\n" + ("ALL PASS" if not FAILURES else f"FAILED: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)
