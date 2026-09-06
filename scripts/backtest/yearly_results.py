"""
Year-by-year results, saved so they can be read back rather than re-run.

Two views of the same trading, because they answer different questions:

  fixed lot   position size never changes, so a year's profit is a clean
              reading of that year's market. This is the view to use when
              comparing 2008 against 2019.

  compounding one continuous run in which size follows the live balance,
              never restarted at a year boundary. This is what the account
              would actually have done, and the view to use when asking
              what the strategy is worth.

The compounding run is deliberately NOT chopped into separate yearly
backtests. Doing that resets the grid every January, closes whatever was
open, and quietly changes the strategy being measured.

    python scripts/backtest/yearly_results.py     ->  results/*.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive import AdaptiveConfig, run_adaptive  # noqa: E402

from quantdata import load_bars  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "baseline"
START, END = "2006-01-01", "2024-12-31"
CASH = 100.0

PARAMS = json.loads((ROOT / "params_100usd.json").read_text())
SPECS = json.loads((ROOT / "symbol_specs.json").read_text())
GRID = {k: PARAMS[k] for k in
        ("atr_mult", "max_open", "stop_extra", "rebound", "vol_block", "vol_floor",
         "atr_fast", "atr_slow", "side", "er_window", "er_block",
         "dd_pause", "dd_pause_bars", "max_hold_days")}
GRID["equity_stop"] = PARAMS["equity_stop"]
MAX_LOT = {"EURUSD": 40.0, "XAUUSD": 30.0}
MARKETS = [("EURUSD", "EURUSD.DUKA"), ("XAUUSD", "XAUUSD.DUKA")]


def money(spec, **extra):
    s = SPECS[spec]
    pip = s["point"] * 10
    return dict(pip=pip, pip_value_per_lot=pip * s["contract"], lot=0.01,
                spread_pips=s["spread_points"] / 10,
                swap_long_pips=s["swap_long"] / 10,
                swap_short_pips=s["swap_short"] / 10,
                start_cash=CASH, max_lot=MAX_LOT[spec], **extra)


def yearly(res, spec, mode):
    eq = CASH + res["equity"]
    ye = eq.resample("YE").last()
    opening = ye.shift(1).fillna(CASH)
    tr = res["trades"].copy()
    tr["y"] = pd.to_datetime(tr["closed_at"]).dt.year
    dd = eq.groupby(eq.index.year).apply(
        lambda s: float(((s.cummax() - s) / s.cummax()).max()) * 100)
    return pd.DataFrame({
        "market": spec, "mode": mode, "year": [t.year for t in ye.index],
        "opening_balance": opening.round(2).values,
        "closing_balance": ye.round(2).values,
        "gain_usd": (ye - opening).round(2).values,
        "gain_pct": ((ye / opening - 1) * 100).round(2).values,
        "max_drawdown_pct_of_peak": [round(dd.get(t.year, np.nan), 2) for t in ye.index],
        "lot_used": [round(tr.loc[tr.y == t.year, "lot"].max(), 3) for t in ye.index],
        "trades": [int((tr.y == t.year).sum()) for t in ye.index],
        "win_pct": [round(float((tr.loc[tr.y == t.year, "net_pips"] > 0).mean() * 100), 1)
                    for t in ye.index],
    })


def summarise(t):
    k = len(t)
    total = t.closing_balance.iloc[-1]
    # `t.mode` would resolve to DataFrame.mode, not the column.
    return dict(market=t["market"].iloc[0], mode=t["mode"].iloc[0], years=k,
                final_balance=round(total, 2),
                cagr_pct=round(((total / CASH) ** (1 / k) - 1) * 100, 2),
                positive_years=int((t.gain_pct > 0).sum()),
                median_year_pct=round(t.gain_pct.median(), 2),
                best_year_pct=round(t.gain_pct.max(), 2),
                worst_year_pct=round(t.gain_pct.min(), 2),
                year_to_year_swing=round(t.gain_pct.std(), 2),
                worst_drawdown_pct=round(t.max_drawdown_pct_of_peak.max(), 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables, summaries = [], []
    for spec, sym in MARKETS:
        df = load_bars(sym, "m5", start=START, end=END)
        for mode, extra in (("fixed_lot", {}), ("compounding", {"compound": True})):
            res = run_adaptive(df, AdaptiveConfig(**GRID, **money(spec, **extra)))
            t = yearly(res, spec, mode)
            tables.append(t)
            summaries.append(summarise(t))
            print(f"  {spec:<8} {mode:<12} ${CASH:.0f} -> "
                  f"${t.closing_balance.iloc[-1]:>9,.0f}  "
                  f"median year {t.gain_pct.median():>6.1f}%  "
                  f"positive {int((t.gain_pct>0).sum())}/{len(t)}")

    pd.concat(tables, ignore_index=True).to_csv(OUT / "yearly_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "yearly_summary.csv", index=False)
    print(f"\nsaved -> {OUT / 'yearly_results.csv'}")
    print(f"saved -> {OUT / 'yearly_summary.csv'}")


if __name__ == "__main__":
    main()
