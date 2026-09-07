"""
The current picture, drawn from scratch after every correction.

Nothing older than this run survives: the previous charts were all made
before at least one of four fixes, and are quarantined in
_invalid_outputs/ rather than shown. What is drawn here uses

  * drawdown measured against the equity it fell from, not the opening $100
  * an equity stop that cannot close a position on its own opening bar
  * true range that does not reach across a hole in the data
  * gold from Dukascopy, 23 years, rather than 1.4 years of MT5

Charts go to plots/. wiki_PNG/ belongs to the user -- it holds reference
images they bring to the conversation, and nothing here writes to it.

    python scripts/backtest/plots_current.py
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
from adaptive import AdaptiveConfig, run_adaptive, volatility  # noqa: E402
from plot_run import draw  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
# BASELINE -- ล็อกแล้ว ห้ามแก้ไฟล์ในโฟลเดอร์นี้
# อ่านค่าจาก configs/baseline.json เท่านั้น และเขียนผลลง plots/baseline/ กับ results/baseline/
# เวอร์ชันอื่นมีสำเนาสคริปต์ของตัวเอง จึงแก้ทับกันไม่ได้
VERSION = "baseline"
OUT = ROOT / "plots" / VERSION
CASH = 100.0

PARAMS = json.loads((ROOT / "configs" / f"{VERSION}.json").read_text())
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
GRID = {k: PARAMS[k] for k in
        ("atr_mult", "max_open", "stop_extra", "rebound", "vol_block", "vol_floor",
         "atr_fast", "atr_slow", "side", "er_window", "er_block",
         "dd_pause", "dd_pause_bars", "max_hold_days")}
GRID["equity_stop"] = PARAMS["equity_stop"]
MAX_LOT = {"EURUSD": 40.0, "XAUUSD": 30.0}

RUNS = [
    ("EURUSD_train", "EURUSD.DUKA", "EURUSD", "2006-01-01", "2024-12-31",
     "EURUSD m5, 2006-2024 (the search saw this)"),
    ("EURUSD_oos",   "EURUSD.DUKA", "EURUSD", "2025-01-01", "2026-12-31",
     "EURUSD m5, 2025-2026 OUT OF SAMPLE"),
    ("GOLD_train",   "XAUUSD.DUKA", "XAUUSD", "2006-01-01", "2024-12-31",
     "Gold m5, 2006-2024 (never used to choose anything)"),
    ("GOLD_oos",     "XAUUSD.DUKA", "XAUUSD", "2025-01-01", "2026-12-31",
     "Gold m5, 2025-2026 OUT OF SAMPLE"),
]


def money(spec):
    s = SPECS[spec]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=MAX_LOT[spec])


def account_panel(res, title, path):
    """Balance, drawdown against its own peak, and profit per month."""
    eq = CASH + res["equity"]
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100
    monthly = res["equity"].resample("ME").last().dropna().diff().dropna()

    fig, ax = plt.subplots(3, 1, figsize=(15, 10),
                           gridspec_kw={"height_ratios": [2, 1, 1.2], "hspace": 0.3})
    ax[0].plot(eq.index, eq, lw=1.1, color="#111")
    ax[0].axhline(CASH, color="grey", ls="--", lw=.8)
    ax[0].fill_between(eq.index, CASH, eq, where=eq >= CASH, color="#2e9e5b", alpha=.18)
    ax[0].fill_between(eq.index, CASH, eq, where=eq < CASH, color="crimson", alpha=.18)
    ax[0].set_ylabel("balance ($), from $100")
    ax[0].set_title(title, fontsize=12)
    ax[0].grid(alpha=.25)

    ax[1].fill_between(dd.index, dd, 0, color="crimson", alpha=.45)
    ax[1].axhline(-30, color="darkred", ls=":", lw=1, label="30% below the peak")
    ax[1].set_ylabel("drawdown (% of peak)")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].grid(alpha=.25)

    ax[2].bar(monthly.index, monthly, width=20,
              color=["#2e9e5b" if v >= 0 else "crimson" for v in monthly], alpha=.85)
    ax[2].axhline(0, color="grey", lw=.8)
    ax[2].set_ylabel("profit per month ($)")
    ax[2].grid(alpha=.25)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return monthly


def regime_chart(store, path):
    """
    Profit per year against how much the market moved that year.

    This is the chart the year-by-year numbers asked for: the strategy is
    paid by volatility and by nothing else the data could find, so the
    quiet years are not bad luck, they are the absence of the thing it
    trades. Both markets go quiet in the same stretch, which is why
    holding more instruments cannot fill those years in.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    for ax, (lbl, (net, vol)) in zip(axes, store.items()):
        yrs = [t.year for t in net.index]
        ax.bar(yrs, net.values, color=["#2e9e5b" if v >= 0 else "crimson" for v in net],
               alpha=.85, label=f"{lbl} profit that year ($)")
        ax.axhline(0, color="grey", lw=.8)
        ax.set_ylabel(f"{lbl} profit ($)")
        ax2 = ax.twinx()
        ax2.plot(yrs, vol.values, color="#3b5bdb", lw=2, marker="o", ms=4,
                 label="annualised volatility (%)")
        ax2.set_ylabel("volatility (%)", color="#3b5bdb")
        ax.grid(alpha=.25)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    axes[0].set_title("Profit arrives with volatility, and both markets go quiet "
                      "in the same years (2013-2019)", fontsize=12)
    axes[-1].set_xlabel("year")
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store, summary = {}, []
    for tag, sym, spec, a, b, label in RUNS:
        df = load_bars(sym, "m5", start=a, end=b)
        kw = money(spec)
        cfg = AdaptiveConfig(**GRID, **kw)
        res = run_adaptive(df, cfg)

        eq = CASH + res["equity"]
        peak = eq.cummax()
        maxdd = float(((peak - eq) / peak).max())
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        head = (f"{label}  |  $100, 0.01 lot, ${GRID['equity_stop']:.0f} equity stop\n"
                f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} ({yrs:.1f}y)   "
                f"$100 -> ${eq.iloc[-1]:,.0f}   worst drawdown {maxdd:.0%} of its peak   "
                f"${res['per_day']:.3f}/day   {res['n_trades']/yrs:,.0f} trades a year")

        monthly = account_panel(res, head, OUT / f"account_{tag}.png")
        draw(df, res, cfg, head, OUT / f"grid_{tag}.png", max_markers=700)

        if tag.endswith("_train"):
            net = res["equity"].resample("YE").last().diff()
            net.iloc[0] = res["equity"].resample("YE").last().iloc[0]
            live = df.high > df.low
            dayc = df["close"][live].resample("1D").last().dropna()
            vol = dayc.pct_change().resample("YE").std() * np.sqrt(252) * 100
            store[label.split(" m5")[0]] = (net, vol.reindex(net.index))

        summary.append((label, yrs, eq.iloc[-1], maxdd, res["per_day"],
                        res["n_trades"], (monthly > 0).sum(), len(monthly)))
        print(f"  {label:<52} $100 -> ${eq.iloc[-1]:>7,.0f}  dd {maxdd:>4.0%}  "
              f"${res['per_day']:.3f}/day  {(monthly>0).sum()}/{len(monthly)} months up")

    regime_chart(store, OUT / "regime_by_year.png")
    print("\nsaved to plots/:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
