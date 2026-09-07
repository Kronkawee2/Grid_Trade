"""
Draw every market the fixed configuration was tested on.

One configuration, carried unchanged across instruments: the only thing
that varies is the broker's own arithmetic for each symbol -- pip size,
contract size, spread and swap, all read from MT5 rather than assumed.
Anything else varying by instrument would make the comparison meaningless,
because a setting tuned per symbol has been fitted to that symbol's
history and can no longer be evidence about the next one.

Produces, in plots/:
    compare_all_markets.png   every equity curve on one pair of axes
    account_<sym>_<tf>.png    balance, drawdown and monthly profit
    grid_<sym>_<tf>.png       price with grid boxes, volatility, equity

    python scripts/backtest/charts_all_markets.py
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
from plot_run import draw  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
# Charts this project generates. wiki_PNG next door belongs to the
# user -- it holds reference images they bring to the conversation,
# and nothing here writes to it.
PNG = ROOT / "plots"
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

LOT = 0.01

PARAMS = json.loads((ROOT / "params_100usd.json").read_text())
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
GRID = {k: PARAMS[k] for k in
        ("atr_mult", "max_open", "stop_extra", "rebound", "vol_block", "vol_floor",
         "atr_fast", "atr_slow", "side", "er_window", "er_block",
         "dd_pause", "dd_pause_bars", "max_hold_days")}
GRID["equity_stop"] = PARAMS["equity_stop"]

MARKETS = [
    ("EURUSD.DUKA", "EURUSD", "m5",  "2006-01-01", "EURUSD m5 (Dukascopy, 20 years)"),
    ("EURUSD",      "EURUSD", "m5",  "2000-01-01", "EURUSD m5 (Eightcap)"),
    ("XAUUSD",      "XAUUSD", "m5",  "2000-01-01", "Gold m5 (Eightcap)"),
    ("XAUUSD",      "XAUUSD", "m15", "2000-01-01", "Gold m15 (Eightcap)"),
    ("XAGUSD",      "XAGUSD", "m5",  "2000-01-01", "Silver m5 (Eightcap)"),
    ("XAGUSD",      "XAGUSD", "m15", "2000-01-01", "Silver m15 (Eightcap)"),
    ("NDX100",      "NDX100", "m5",  "2000-01-01", "Nasdaq m5 (Eightcap)"),
    ("NDX100",      "NDX100", "m15", "2000-01-01", "Nasdaq m15 (Eightcap)"),
]


def money(spec_name):
    """MT5 quotes swap and spread in points; one pip is ten of them."""
    s = SPECS[spec_name]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=LOT,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10)


def account_panel(res, title, path):
    eq = CASH + res["equity"]
    eq_peak = (CASH + res["equity"]).cummax()
    dd_pct = res["drawdown"] / eq_peak * 100
    monthly = res["equity"].resample("ME").last().dropna().diff().dropna()

    fig, ax = plt.subplots(3, 1, figsize=(15, 10),
                           gridspec_kw={"height_ratios": [2, 1, 1.2], "hspace": 0.3})
    ax[0].plot(eq.index, eq, lw=1.1, color="#111")
    ax[0].axhline(CASH, color="grey", ls="--", lw=.8)
    ax[0].fill_between(eq.index, CASH, eq, where=eq >= CASH, color="#2e9e5b", alpha=.18)
    ax[0].fill_between(eq.index, CASH, eq, where=eq < CASH, color="crimson", alpha=.18)
    ax[0].set_ylabel("balance ($)")
    ax[0].set_title(title, fontsize=12)
    ax[0].grid(alpha=.25)

    ax[1].fill_between(dd_pct.index, dd_pct, 0, color="crimson", alpha=.45)
    ax[1].axhline(-100, color="black", ls="-", lw=1.2, label="the whole $100")
    ax[1].set_ylabel("drawdown (% of $100)")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].grid(alpha=.25)

    ax[2].bar(monthly.index, monthly, width=20,
              color=["#2e9e5b" if v >= 0 else "crimson" for v in monthly], alpha=.85)
    ax[2].axhline(0, color="grey", lw=.8)
    ax[2].set_ylabel("profit per month ($)")
    ax[2].grid(alpha=.25)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    results, rows = {}, []
    for sym, spec, tf, start, label in MARKETS:
        try:
            df = load_bars(sym, tf, start=start, end="2026-12-31")
        except Exception as e:
            print(f"  skip {sym} {tf}: {e}")
            continue
        if len(df) < 5000:
            print(f"  skip {sym} {tf}: only {len(df):,} bars")
            continue

        kw = money(spec)
        cfg = AdaptiveConfig(**GRID, **kw)
        res = run_adaptive(df, cfg)
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        monthly = res["equity"].resample("ME").last().dropna().diff().dropna()
        dd = abs(res["max_drawdown"])
        st = drawdown_stats(res, CASH)
        survived = not st["ruined"]

        tag = f"{sym.replace('.', '_')}_{tf}"
        head = (f"{label}  |  one fixed configuration, 0.01 lot, $100 account\n"
                f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} ({yrs:.1f}y)   "
                f"net ${res['net_profit']:,.0f}   worst drawdown ${res['max_drawdown']:,.0f} "
                f"({st['max_dd_pct']:.0%} from its peak"
                f"{'' if survived else ', ACCOUNT RUINED'})   "
                f"${res['per_day']:.3f}/day   {(monthly > 0).sum()}/{len(monthly)} months up")
        account_panel(res, head, PNG / f"account_{tag}.png")
        draw(df, res, cfg, head, PNG / f"grid_{tag}.png", max_markers=600)

        results[label] = (CASH + res["equity"], survived)
        rows.append((label, yrs, res["net_profit"], res["max_drawdown"], st["max_dd_pct"],
                     res["per_day"], (monthly > 0).sum(), len(monthly)))
        print(f"  {label:<34} ${st['final']:>7,.0f}  dd {st['max_dd_pct']:>4.0%}  "
              f"${res['per_day']:.3f}/day  {(monthly>0).sum()}/{len(monthly)} months")

    # --- one comparison sheet ------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(15, 11),
                           gridspec_kw={"height_ratios": [2, 1], "hspace": 0.28})
    cmap = plt.get_cmap("tab10")
    for i, (label, (eq, survived)) in enumerate(results.items()):
        ax[0].plot(eq.index, eq, lw=1.3, color=cmap(i % 10),
                   ls="-" if survived else ":",
                   label=label + ("" if survived else "  (account gone)"))
    ax[0].axhline(CASH, color="grey", ls="--", lw=1)
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_ylabel("balance ($), all starting at $100")
    ax[0].set_title("One configuration, every market. Dotted = drawdown exceeded "
                    "the $100 account at some point.", fontsize=12)
    ax[0].legend(fontsize=9, ncol=2)
    ax[0].grid(alpha=.25)

    labels = [r[0] for r in rows]
    share = [r[6] / r[7] * 100 for r in rows]
    ddp = [r[4] * 100 for r in rows]
    x = np.arange(len(labels))
    ax[1].bar(x - .2, share, .4, label="% of months profitable", color="#2e9e5b")
    ax[1].bar(x + .2, ddp, .4, label="worst drawdown, % of $100", color="crimson")
    ax[1].axhline(50, color="#2e9e5b", ls=":", lw=1)
    ax[1].axhline(100, color="crimson", ls=":", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=.25, axis="y")
    fig.savefig(PNG / "compare_all_markets.png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {len(rows)*2 + 1} charts to {PNG}")


if __name__ == "__main__":
    main()
