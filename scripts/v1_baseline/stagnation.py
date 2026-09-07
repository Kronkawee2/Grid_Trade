"""
Why the equity curve is a staircase instead of a curve.

Compounding should bend a chart upward. These do not: they climb hard for
a few months and then sit still for a year or more. This measures the
sitting still -- how long it lasts, how much of the record it accounts
for, and what the market was doing at the time -- because a strategy that
is idle for two thirds of its life is a different proposition from one
that compounds steadily, whatever the two of them average out to.

The measure is days since the account last made a new high. That is the
only definition that matches what the account holder experiences: the
balance has been seen before, and nothing since has beaten it.

    python scripts/backtest/stagnation.py   ->  results/, plots/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
# BASELINE -- ล็อกแล้ว ห้ามแก้ไฟล์ในโฟลเดอร์นี้
# อ่านค่าจาก configs/v1_baseline.json เท่านั้น และเขียนผลลง plots/v1_baseline/ กับ results/v1_baseline/
# เวอร์ชันอื่นมีสำเนาสคริปต์ของตัวเอง จึงแก้ทับกันไม่ได้
VERSION = "v1_baseline"
RESULTS = ROOT / "results" / VERSION
PLOTS = ROOT / "plots" / ("experiments" if "--compound" in sys.argv
                          else VERSION)
START, END, CASH = "2006-01-01", "2024-12-31", 100.0

# A stretch shorter than this is an ordinary pause between trades, not a
# stagnation worth naming.
MIN_FLAT_DAYS = 90

# The baseline trades a fixed 0.01 lot. Compounding is a separate
# experiment and mixing the two in one chart makes the picture unreadable,
# so it is a switch rather than a default.
COMPOUND = "--compound" in sys.argv
TAG = "compounding" if COMPOUND else "baseline"

PARAMS = json.loads((ROOT / "configs" / f"{VERSION}.json").read_text())
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
GRID = {k: PARAMS[k] for k in
        ("atr_mult", "max_open", "stop_extra", "rebound", "vol_block", "vol_floor",
         "atr_fast", "atr_slow", "side", "er_window", "er_block",
         "dd_pause", "dd_pause_bars", "max_hold_days")}
GRID["equity_stop"] = PARAMS["equity_stop"]
MAX_LOT = {"EURUSD": 40.0, "XAUUSD": 30.0}
MARKETS = [("EURUSD", "EURUSD.DUKA"), ("XAUUSD", "XAUUSD.DUKA")]


def money(spec):
    s = SPECS[spec]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=MAX_LOT[spec], compound=COMPOUND)


def flat_episodes(daily_eq):
    """Every stretch between one equity high and the next that beats it."""
    peak = daily_eq.cummax()
    at_high = daily_eq >= peak - 1e-9
    highs = list(daily_eq.index[at_high])
    out = []
    for a, b in zip(highs, highs[1:]):
        days = (b - a).days
        if days >= MIN_FLAT_DAYS:
            out.append(dict(start=a, end=b, days=days,
                            balance=round(float(daily_eq[a]), 2)))
    # the run may simply end without recovering
    if highs and (daily_eq.index[-1] - highs[-1]).days >= MIN_FLAT_DAYS:
        out.append(dict(start=highs[-1], end=daily_eq.index[-1],
                        days=(daily_eq.index[-1] - highs[-1]).days,
                        balance=round(float(daily_eq[highs[-1]]), 2)))
    return pd.DataFrame(out)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    all_ep, summary, curves = [], [], {}

    for spec, sym in MARKETS:
        df = load_bars(sym, "m5", start=START, end=END)
        res = run_adaptive(df, AdaptiveConfig(**GRID, **money(spec)))
        eq = (CASH + res["equity"]).resample("1D").last().dropna()

        # realised volatility of the market itself, for context
        live = df.high > df.low
        dayc = df["close"][live].resample("1D").last().dropna()
        vol = dayc.pct_change().rolling(60).std() * np.sqrt(252) * 100

        ep = flat_episodes(eq)
        if len(ep):
            ep["market"] = spec
            ep["years"] = (ep["days"] / 365.25).round(2)
            ep["market_vol_pct"] = [round(float(vol.loc[a:b].mean()), 1)
                                    for a, b in zip(ep.start, ep.end)]
            all_ep.append(ep)

        days_flat = int(ep["days"].sum()) if len(ep) else 0
        span = (eq.index[-1] - eq.index[0]).days
        growing_vol = float(vol[eq.pct_change(20) > 0.02].mean())
        flat_vol = float(ep["market_vol_pct"].mean()) if len(ep) else np.nan
        summary.append(dict(
            market=spec, span_days=span, days_in_stagnation=days_flat,
            share_of_life_stagnant_pct=round(days_flat / span * 100, 1),
            episodes=len(ep),
            longest_flat_days=int(ep["days"].max()) if len(ep) else 0,
            longest_flat_years=round(ep["days"].max() / 365.25, 2) if len(ep) else 0,
            median_flat_days=int(ep["days"].median()) if len(ep) else 0,
            market_vol_when_flat=round(flat_vol, 1) if len(ep) else np.nan,
            market_vol_when_growing=round(growing_vol, 1)))
        curves[spec] = (eq, ep, vol)

        print(f"\n{spec}: {days_flat:,} of {span:,} days stuck "
              f"({days_flat/span:.0%} of the record), {len(ep)} episodes")
        for _, e in ep.iterrows():
            print(f"   {e.start:%Y-%m} -> {e['end']:%Y-%m}  {e.days:>4} days "
                  f"({e.years:.1f}y)  balance stuck at ${e.balance:>9,.0f}  "
                  f"market vol {e.market_vol_pct:.1f}%")

    if all_ep:
        pd.concat(all_ep, ignore_index=True).to_csv(
            RESULTS / f"stagnation_episodes_{TAG}.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS / f"stagnation_summary_{TAG}.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"hspace": 0.25})
    for ax, (spec, (eq, ep, vol)) in zip(axes, curves.items()):
        ax.semilogy(eq.index, eq, lw=1.3, color="#111", label=f"{spec} balance (log)")
        for _, e in ep.iterrows():
            ax.axvspan(e.start, e["end"], color="crimson", alpha=.15)
        ax.set_ylabel("balance ($, log scale)")
        ax.set_title(f"{spec}: shaded = no new equity high for {MIN_FLAT_DAYS}+ days "
                     f"({int(ep['days'].sum()) if len(ep) else 0} days, "
                     f"{len(ep)} episodes)", fontsize=11)
        ax.grid(alpha=.25, which="both")
        ax.legend(loc="upper left", fontsize=9)
    fig.savefig(PLOTS / f"stagnation_{TAG}.png", dpi=125, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved -> {RESULTS/f'stagnation_episodes_{TAG}.csv'}")
    print(f"saved -> {RESULTS/f'stagnation_summary_{TAG}.csv'}")
    print(f"saved -> {PLOTS/f'stagnation_{TAG}.png'}")


if __name__ == "__main__":
    main()
