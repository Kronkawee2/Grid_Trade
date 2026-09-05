"""
Regenerate the charts for the configuration actually being proposed.

Every earlier picture in wiki_PNG was drawn for a setting that has since
been superseded -- a grid with no equity stop, or sized for an account
that is not the one being traded. Those are in plots/superseded now.
Keeping them on display would have been the most expensive kind of stale
documentation: charts that look authoritative and describe something no
longer true.

    python scripts/backtest/final_charts.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402
from plot_run import draw  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
# Charts this project generates. wiki_PNG next door belongs to the
# user -- it holds reference images they bring to the conversation,
# and nothing here writes to it.
PNG = ROOT / "plots"
SYMBOL, TIMEFRAME = "EURUSD.DUKA", "m5"
CASH = 100.0

def drawdown_stats(res, start_cash):
    """
    Drawdown as a share of the equity it actually fell from.

    Dividing the dollar drawdown by the STARTING balance is the mistake
    this replaces. It reads correctly only while the account has not
    grown: once it had, a $114 fall from a $540 peak was reported as
    "114% of the account -- gone", when it was 21% and the run finished at
    $440. The error is always pessimistic and grows with the result, so
    the better a configuration did, the worse it was made to look -- and
    on that basis gold m15 was written off as unusable when it was the
    steadiest market in the set.
    """
    eq = start_cash + res["equity"]
    peak = eq.cummax()
    return {
        "final": float(eq.iloc[-1]),
        "max_dd_pct": float(((peak - eq) / peak).max()),
        "max_dd_cash": float(res["max_drawdown"]),
        "ruined": bool(eq.min() <= 0),
    }


# The proposed live configuration. Spread is set to 1.8 pips rather than
# the 1.0 Eightcap quotes right now, because a live grid fills during news
# and a quoted average is not what it pays then.
PARAMS = dict(
    atr_mult=4.2, max_open=3, stop_extra=1.5, rebound=0.45,
    vol_block=1.15, vol_floor=0.0, atr_fast=20, atr_slow=200,
    side="both", er_window=0, er_block=1.0,
    dd_pause=None, dd_pause_bars=350, max_hold_days=None,
    equity_stop=22.0, spread_pips=1.8, lot=0.01,
)


def account_panel(res, name, path):
    """Equity as an account balance, which is the view that decides things."""
    eq = CASH + res["equity"]
    eq_peak = (CASH + res["equity"]).cummax()
    dd_pct = res["drawdown"] / eq_peak * 100
    daily = res["equity"].resample("1D").last().dropna().diff().dropna()
    monthly = res["equity"].resample("ME").last().dropna().diff().dropna()

    fig, ax = plt.subplots(3, 1, figsize=(15, 10),
                           gridspec_kw={"height_ratios": [2, 1, 1.2], "hspace": 0.28})

    ax[0].plot(eq.index, eq, lw=1.1, color="#111")
    ax[0].axhline(CASH, color="grey", ls="--", lw=0.8)
    ax[0].fill_between(eq.index, CASH, eq, where=eq >= CASH, color="#2e9e5b", alpha=.18)
    ax[0].fill_between(eq.index, CASH, eq, where=eq < CASH, color="crimson", alpha=.18)
    ax[0].set_ylabel("account balance ($)")
    ax[0].set_title(name, fontsize=12)
    ax[0].grid(alpha=.25)

    ax[1].fill_between(dd_pct.index, dd_pct, 0, color="crimson", alpha=.45)
    ax[1].axhline(-30, color="darkred", ls=":", lw=1,
                  label="30% of the account")
    ax[1].set_ylabel("drawdown (% of $100)")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].grid(alpha=.25)

    colours = ["#2e9e5b" if v >= 0 else "crimson" for v in monthly]
    ax[2].bar(monthly.index, monthly, width=20, color=colours, alpha=.85)
    ax[2].axhline(0, color="grey", lw=.8)
    ax[2].set_ylabel("profit per month ($)")
    ax[2].grid(alpha=.25)

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return daily, monthly


def main():
    cfg = AdaptiveConfig(**PARAMS)
    (ROOT / "params_100usd.json").write_text(json.dumps(PARAMS, indent=2))

    runs = {}
    for tag, a, b in [("train_2006_2024", "2006-01-01", "2024-12-31"),
                      ("oos_2025_2026", "2025-01-01", "2026-12-31")]:
        df = load_bars(SYMBOL, TIMEFRAME, start=a, end=b)
        res = run_adaptive(df, cfg)
        runs[tag] = (df, res)
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        seen = "the search saw this" if "train" in tag else "NEVER SEEN"

        head = (f"{SYMBOL} {TIMEFRAME}  |  $100 account, 0.01 lot, "
                f"{PARAMS['spread_pips']}p spread, ${PARAMS['equity_stop']:.0f} equity stop  |  "
                f"{a} -> {df.index[-1]:%Y-%m-%d} ({seen})\n"
                f"net ${res['net_profit']:,.0f}   worst drawdown "
                f"${res['max_drawdown']:,.0f} ({drawdown_stats(res, CASH)['max_dd_pct']:.0%} from its peak)   "
                f"${res['per_day']:.3f}/day   {res['n_trades']/yrs:,.0f} trades a year")

        draw(df, res, cfg, head, PNG / f"grid_{tag}.png", max_markers=700)
        daily, monthly = account_panel(res, head, PNG / f"account_{tag}.png")

        print(f"\n--- {tag} ---")
        print(f"  net ${res['net_profit']:,.0f} | worst DD ${res['max_drawdown']:,.0f} "
              f"({drawdown_stats(res, CASH)['max_dd_pct']:.0%} from peak) | ${res['per_day']:.3f}/day")
        print(f"  {res['n_trades']:,} trades ({res['n_trades']/yrs:,.0f}/yr) | "
              f"win {res['win_rate']:.1f}% | {res['n_equity_stops']:,} equity stops")
        print(f"  months: {(monthly > 0).sum()}/{len(monthly)} profitable | "
              f"median ${monthly.median():.2f} | worst ${monthly.min():.2f}")
        print(f"  worst day ${daily.min():.2f} | best day ${daily.max():.2f}")

    print("\nsaved to plots/:")
    for p in sorted(PNG.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
