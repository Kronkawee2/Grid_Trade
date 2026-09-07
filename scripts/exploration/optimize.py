"""
Search the adaptive grid's parameters on the training years, and draw what
the search actually found.

The picture matters as much as the winner. A search that has found a real
effect draws a cloud with a shape: scores rise toward some region of the
parameter and fall away from it. A search that has found nothing draws a
formless scatter, and its "best" point is the luckiest draw rather than
the right answer -- which is exactly what the reference optimisation
report turned out to be showing.

So every trial is plotted, one panel per parameter, and the winner is
marked. Read the shape before reading the number.

    python scripts/backtest/optimize.py --trials 300

Writes results_optimize.csv and plots/optimization_landscape.png.
"""

import argparse
import sys
import time
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
SYMBOL = "EURUSD.DUKA"
TRAIN_START, TRAIN_END = "2006-01-01", "2024-12-31"
SPREAD_PIPS = 1.0

# Ranges, not points. A range that the search never pushes against is a
# range that was wide enough; a winner sitting on an edge means the space
# was cut off in the wrong place and should be widened before believing it.
#
# `side` is deliberately absent. When it was searchable the winner came
# back sell-only, which is not a grid setting at all -- it is a memory of
# the fact that EURUSD fell from 1.18 to 1.04 across the training years.
# The following two years it rose 1,400 pips and that configuration lost
# on every one of its 68 trades' worth of exposure. A grid is supposed to
# earn from oscillation; letting the search pick a direction lets it earn
# from hindsight instead, and hindsight does not renew.
SPACE = {
    # Widened after the first m5 search put its winner on three separate
    # boundaries -- stop_extra at the low edge, atr_fast and atr_slow both
    # at the high edge. A winner on an edge is not a winner; it is the
    # search saying the answer may be outside the box it was given.
    "atr_mult":      ("float", 4.0, 14.0),
    "max_open":      ("choice", [2, 3]),        # the user's cap: 2-3 positions
    "stop_extra":    ("float", 0.3, 6.0),
    "vol_block":     ("float", 1.0, 3.5),
    "atr_fast":      ("choice", [10, 20, 50, 100, 200]),
    "atr_slow":      ("choice", [100, 200, 500, 1000, 2000]),
    # Trend gate: 0 disables it, so the search decides whether it earns
    # its place rather than being told it has one.
    "er_window":     ("choice", [0, 30, 60, 120, 240]),
    "er_block":      ("float", 0.10, 0.70),
    # Circuit breaker, in account dollars from the equity peak.
    "dd_pause":      ("choice", [None, 20, 40, 80]),
    "dd_pause_bars": ("choice", [300, 1000, 3000]),
    "max_hold_days": ("choice", [None, 5, 20, 60]),
}
FIXED = {"side": "both"}

# A configuration that trades a handful of times over nineteen years has
# not been tested by them. Its score is an accident of which handful.
MIN_TRADES = 200


def sample(rng):
    out = {}
    for k, spec in SPACE.items():
        if spec[0] == "float":
            out[k] = float(rng.uniform(spec[1], spec[2]))
        else:
            choices = spec[1]
            out[k] = choices[rng.integers(len(choices))]
    return out


def yearly_net(res) -> pd.Series:
    eq = res["equity"].resample("YE").last()
    return eq.diff().fillna(eq.iloc[0] if len(eq) else 0.0)


def objective(res):
    """
    Profit per unit of worst drawdown, scaled by how many years actually
    contributed.

    The drawdown term is there because an account that can be closed by a
    drawdown cannot spend a return it did not survive to collect: $500
    earned while $3,000 under water ranks below $400 earned while never
    more than $200 down. Lot size can be multiplied afterwards; a blown
    account cannot.

    The consistency term is there because a single nineteen-year total
    hides its own shape. A configuration that made everything in the 2008
    and 2011 crises and drifted sideways for the next decade scores the
    same as one that earned a little every year, and only the second is
    something to run next month. Multiplying by the share of profitable
    years makes the sideways decade cost what it should.
    """
    dd = abs(res["max_drawdown"])
    if res["n_trades"] < MIN_TRADES or dd == 0:
        return -np.inf
    yr = yearly_net(res)
    consistency = float((yr > 0).mean()) if len(yr) else 0.0
    return res["net_profit"] / dd * consistency


def evaluate(df, params):
    cfg = AdaptiveConfig(spread_pips=SPREAD_PIPS, **params, **FIXED)
    res = run_adaptive(df, cfg)
    yr = yearly_net(res)
    return res, {
        **params,
        "score": objective(res),
        "net": res["net_profit"],
        "dd": res["max_drawdown"],
        "per_day": res["per_day"],
        "good_years": int((yr > 0).sum()),
        "n_years": len(yr),
        "worst_year": float(yr.min()) if len(yr) else np.nan,
        "median_year": float(yr.median()) if len(yr) else np.nan,
        "trades": res["n_trades"],
        "stops": res["n_stops"],
        "win_rate": res["win_rate"],
        "peak_open": res["max_concurrent"],
    }


def plot_landscape(df: pd.DataFrame, best: pd.Series, path: Path, timeframe: str):
    scored = df[np.isfinite(df["score"])]
    params = [k for k in SPACE if scored[k].nunique() > 1]
    ncols = 4
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, p in zip(axes, params):
        x = scored[p]
        if x.dtype == object or isinstance(x.iloc[0], (str, type(None))):
            labels = sorted(x.astype(str).unique())
            xs = x.astype(str).map({v: i for i, v in enumerate(labels)})
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20)
            bx = labels.index(str(best[p]))
        else:
            xs, bx = x, best[p]
        ax.scatter(xs, scored["score"], s=14, alpha=0.45,
                   c=scored["score"], cmap="viridis")
        ax.scatter([bx], [best["score"]], s=160, marker="*",
                   color="crimson", zorder=5, label="best")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(p)
        ax.set_ylabel("net / |max drawdown|")
        ax.grid(alpha=0.25)
    for ax in axes[len(params):]:
        ax.axis("off")

    fig.suptitle(
        f"Adaptive grid optimisation landscape  |  {SYMBOL} {timeframe}  "
        f"{TRAIN_START[:4]}-{TRAIN_END[:4]}  |  {len(scored)} scored trials\n"
        "a real effect looks like a trend; a formless cloud means the best "
        "point is only the luckiest one",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeframe", default="m5")
    args = ap.parse_args()
    TIMEFRAME = args.timeframe

    df = load_bars(SYMBOL, TIMEFRAME, start=TRAIN_START, end=TRAIN_END)
    print(f"{SYMBOL} {TIMEFRAME}: {len(df):,} bars "
          f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    print(f"random search, {args.trials} trials, spread {SPREAD_PIPS}p flat\n")

    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for n in range(1, args.trials + 1):
        _, row = evaluate(df, sample(rng))
        rows.append(row)
        if n % 25 == 0 or n == args.trials:
            done = pd.DataFrame(rows)
            live = done[np.isfinite(done["score"])]
            b = live.loc[live["score"].idxmax()] if len(live) else None
            el = time.time() - t0
            msg = (f"  {n:>4}/{args.trials} | {el/n:.1f}s each | "
                   f"eta {(args.trials-n)*el/n/60:.0f}m | scored {len(live)}")
            if b is not None:
                msg += (f" | best {b['score']:.2f} "
                        f"(net ${b['net']:,.0f}, dd ${b['dd']:,.0f}, "
                        f"{b['good_years']:.0f}/{b['n_years']:.0f} yrs)")
            print(msg, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"results_optimize_{TIMEFRAME}.csv", index=False)

    live = out[np.isfinite(out["score"])]
    if live.empty:
        print("\nNo trial met the minimum trade count -- widen the space.")
        return
    best = live.loc[live["score"].idxmax()]

    png = ROOT / "plots" / f"optimization_landscape_{TIMEFRAME}.png"
    plot_landscape(out, best, png, TIMEFRAME)

    print("\n--- best on training data ---")
    for k in SPACE:
        print(f"  {k:14}: {best[k]}")
    print(f"\n  score (net/DD) : {best['score']:.2f}")
    print(f"  net            : ${best['net']:,.0f}")
    print(f"  max drawdown   : ${best['dd']:,.0f}")
    print(f"  per day        : ${best['per_day']:.2f}")
    print(f"  trades         : {best['trades']:,.0f}   win {best['win_rate']:.1f}%")
    print(f"  peak open      : {best['peak_open']:.0f}")
    print(f"  profitable yrs : {best['good_years']:.0f}/{best['n_years']:.0f}")
    print(f"  median year    : ${best['median_year']:,.0f}   "
          f"worst ${best['worst_year']:,.0f}")

    print("\n--- how sensitive is that peak ---")
    top = live.nlargest(max(5, len(live) // 20), "score")
    for k in SPACE:
        if top[k].dtype == object or isinstance(top[k].iloc[0], (str, type(None))):
            print(f"  {k:14}: " + ", ".join(
                f"{v}={c}" for v, c in top[k].astype(str).value_counts().items()))
        else:
            print(f"  {k:14}: {top[k].min():.2f} .. {top[k].max():.2f} "
                  f"(median {top[k].median():.2f})")

    print(f"\nsaved -> results_optimize.csv")
    print(f"saved -> {png}")


if __name__ == "__main__":
    main()
