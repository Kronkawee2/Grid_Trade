"""
Grid backtest engine. No ML, no regime filter -- deliberately.

This is the baseline everything else gets measured against. Without it
there is no way to tell whether a regime model added anything or merely
added complexity, and in this field the honest answer is usually the
latter. So the first number worth having is what a plain fixed grid does
over twenty years.

Mechanics
---------
Buy levels sit at anchor - 1*S, -2*S ... -N*S; sell levels mirror them
above. Touching a level opens one position there, and that position closes
one grid step in its favour.

The anchor moves only when the grid is flat AND price has travelled
outside the grid's own zone -- the point past which no level is reachable
anyway. While anything is open the anchor is frozen, so a market that
walks away leaves the grid holding its positions until price returns or
the stop fires. That freeze is the real risk of a grid rather than a
defect, so it is modelled rather than engineered away.

This rule was chosen for the baseline because it needs no parameter of its
own; the zone is already defined by spacing and level count. Extending the
grid when price leaves the zone, or re-anchoring on a calendar, would each
introduce a number to tune -- and a baseline whose numbers are tuned is
not a baseline.

Bar ambiguity
-------------
A bar's high and low carry no ordering, so a position may not close on the
bar that opened it. Allowing that books a full grid step of profit out of
a high that may well have come before the entry -- free money invented by
the data format. An earlier version of this file did allow it, and on the
real EURUSD history 10% of trades were same-bar round trips contributing
28% of the headline profit.

Where an entry level and an open position's take profit fall in the same
bar, the entry is assumed first: it books the new exposure and delays the
profit, which is the pessimistic reading.

Costs
-----
Spread comes from the bar's own recorded spread rather than a constant,
because a constant is wrong exactly where it matters. In this dataset
EURUSD's spread has a median near 1 pip but reaches 13+ during stress,
and a grid takes many small profits, so it is more cost-sensitive than
almost any other strategy shape.

Swap is charged per night a position is held, and is asymmetric: on this
broker the long side pays and the short side collects. That asymmetry is
large enough to change which side of the grid is worth running at all,
so it is modelled rather than averaged away.
"""

import numpy as np
import pandas as pd


class GridConfig:
    """
    One grid's rules. Prices are in the instrument's own units; anything
    named *_pips is in pips and converted with `pip` on the way in.
    """

    def __init__(self, spacing_pips=42.0, n_levels=10, stop_pips=None,
                 side="both", lot=0.01, pip=0.0001, pip_value=1.0,
                 swap_long_pips=-0.7, swap_short_pips=0.25,
                 spread_override_pips=None):
        self.spacing = spacing_pips * pip
        self.n_levels = n_levels
        # The default stop must sit strictly beyond the outermost level.
        # Setting it equal to the zone -- spacing * n_levels -- puts it on
        # top of the deepest entry, so the bar that fills the last level
        # also triggers the stop and the grid is flattened before it can
        # work. That produced 268 trades in nineteen years on the first
        # run, which is what surfaced the mistake.
        self.stop = (stop_pips * pip) if stop_pips is not None else self.spacing * n_levels * 2
        self.side = side
        self.lot = lot
        self.pip = pip
        # Account currency per pip for `lot` -- 0.01 lot of EURUSD moves
        # $0.10 per pip, so pip_value=1.0 with lot=0.01 gives 0.10.
        self.pip_value = pip_value * lot * 10
        self.swap_long = swap_long_pips
        self.swap_short = swap_short_pips
        self.spread_override = spread_override_pips

    @property
    def zone_pips(self):
        return self.spacing * self.n_levels / self.pip


class Position:
    __slots__ = ("side", "entry", "tp", "opened_at", "opened_day",
                 "opened_i", "entry_cost_pips")

    def __init__(self, side, entry, tp, opened_at, opened_day, opened_i,
                 entry_cost_pips):
        self.side = side
        self.entry = entry
        self.tp = tp
        self.opened_at = opened_at
        self.opened_day = opened_day
        self.opened_i = opened_i          # bar index, to forbid a same-bar exit
        self.entry_cost_pips = entry_cost_pips


def run_grid(df: pd.DataFrame, cfg: GridConfig) -> dict:
    """
    Run one grid configuration over a bar frame.

    `df` needs high/low/close and, ideally, spread_avg (in points). Returns
    a dict of results including the equity curve, so a caller can compute
    whatever risk measure it wants rather than being handed a fixed set.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    times = df.index.to_numpy()
    days = df.index.normalize().to_numpy()

    if cfg.spread_override is not None:
        spread_pips = np.full(len(df), float(cfg.spread_override))
    elif "spread_avg" in df.columns:
        # stored in points; 10 points = 1 pip on a 5-digit quote
        spread_pips = (df["spread_avg"].fillna(10.0).to_numpy(dtype=float)) / 10.0
    else:
        spread_pips = np.full(len(df), 1.0)

    do_buy = cfg.side in ("buy", "both")
    do_sell = cfg.side in ("sell", "both")

    anchor = close[0]
    open_pos: list[Position] = []
    filled: set[tuple[str, int]] = set()   # levels currently held this cycle

    realised = 0.0
    equity_curve = np.empty(len(df))
    trades = []
    n_stops = 0
    n_resets = 0
    swap_paid = 0.0
    spread_paid = 0.0

    def close_position(p: Position, exit_price: float, t, reason: str):
        nonlocal realised, swap_paid, spread_paid
        direction = 1.0 if p.side == "buy" else -1.0
        gross_pips = (exit_price - p.entry) / cfg.pip * direction
        nights = int((t - p.opened_day) / np.timedelta64(1, "D"))
        swap_pips = nights * (cfg.swap_long if p.side == "buy" else cfg.swap_short)
        net_pips = gross_pips - p.entry_cost_pips + swap_pips
        realised += net_pips * cfg.pip_value
        swap_paid += swap_pips * cfg.pip_value
        spread_paid += p.entry_cost_pips * cfg.pip_value
        trades.append({
            "side": p.side, "entry": p.entry, "exit": exit_price,
            "opened_at": p.opened_at, "closed_at": t, "reason": reason,
            "gross_pips": gross_pips, "cost_pips": p.entry_cost_pips,
            "swap_pips": swap_pips, "net_pips": net_pips,
        })

    def reset(price):
        nonlocal anchor
        anchor = price
        filled.clear()

    for i in range(len(df)):
        t, day = times[i], days[i]
        hi, lo, cl, sp = high[i], low[i], close[i], spread_pips[i]

        # --- stop: price has left the zone far enough to give up ---------
        if open_pos:
            if lo <= anchor - cfg.stop or hi >= anchor + cfg.stop:
                exit_price = anchor - cfg.stop if lo <= anchor - cfg.stop else anchor + cfg.stop
                for p in open_pos:
                    close_position(p, exit_price, t, "stop")
                open_pos.clear()
                n_stops += 1
                reset(cl)
                equity_curve[i] = realised
                continue

        # --- entries (assumed to happen before exits within a bar) -------
        for k in range(1, cfg.n_levels + 1):
            if do_buy:
                lvl = anchor - k * cfg.spacing
                if ("buy", k) not in filled and lo <= lvl:
                    filled.add(("buy", k))
                    open_pos.append(Position("buy", lvl, lvl + cfg.spacing, t, day, i, sp))
            if do_sell:
                lvl = anchor + k * cfg.spacing
                if ("sell", k) not in filled and hi >= lvl:
                    filled.add(("sell", k))
                    open_pos.append(Position("sell", lvl, lvl - cfg.spacing, t, day, i, sp))

        # --- take profits ------------------------------------------------
        still_open = []
        for p in open_pos:
            hit = p.opened_i < i and (
                (p.side == "buy" and hi >= p.tp) or (p.side == "sell" and lo <= p.tp))
            if hit:
                close_position(p, p.tp, t, "tp")
                # free the level so the grid can trade it again this cycle
                lvl_k = int(round(abs(p.entry - anchor) / cfg.spacing))
                filled.discard((p.side, lvl_k))
            else:
                still_open.append(p)
        open_pos = still_open

        # Re-anchor only when the grid is both flat and out of reach of the
        # price -- outside its own zone. Two wrong versions came before it.
        # The first required `filled` to be non-empty while `open_pos` was
        # empty, which cannot happen (a level leaves `filled` at the moment
        # its position leaves `open_pos`), so the grid never moved at all:
        # it held its January-2006 anchor for nineteen years while price
        # travelled 1.18 -> 1.60 -> 0.95. The second re-anchored on every
        # flat bar, which is worse in the opposite direction -- the anchor
        # runs downhill alongside a drifting market and no level is ever
        # reached, so a 1000-pip decline produced no trades whatsoever.
        #
        # The zone is the natural boundary and costs no new parameter: while
        # price is inside it the grid is still doing its job, and once price
        # is beyond it every level is unreachable anyway.
        if not open_pos and abs(cl - anchor) > cfg.spacing * cfg.n_levels:
            reset(cl)
            n_resets += 1

        # Equity marks open positions to market so drawdown is what the
        # account would actually have shown, not just closed profit.
        floating = 0.0
        for p in open_pos:
            direction = 1.0 if p.side == "buy" else -1.0
            floating += ((cl - p.entry) / cfg.pip * direction - p.entry_cost_pips) * cfg.pip_value
        equity_curve[i] = realised + floating

    eq = pd.Series(equity_curve, index=df.index)
    peak = eq.cummax()
    dd = eq - peak

    return {
        "equity": eq,
        "trades": pd.DataFrame(trades),
        "net_profit": realised,
        "max_drawdown": float(dd.min()),
        "n_trades": len(trades),
        "n_stops": n_stops,
        "n_resets": n_resets,
        "swap_paid": swap_paid,
        "spread_paid": spread_paid,
        "open_at_end": len(open_pos),
    }


def summarise(res: dict, years: float, start_cash: float = 100.0) -> dict:
    """Headline numbers, chosen for a strategy that wins often and loses rarely."""
    tr = res["trades"]
    eq = res["equity"]
    daily = eq.resample("1D").last().dropna().diff().dropna()

    wins = (tr["net_pips"] > 0).sum() if len(tr) else 0
    downside = daily[daily < 0]
    sortino = (daily.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else np.nan
    dd = abs(res["max_drawdown"])

    return {
        "net_profit": res["net_profit"],
        "return_pct": res["net_profit"] / start_cash * 100,
        "max_drawdown": -dd,
        "dd_pct_of_start": -dd / start_cash * 100,
        "calmar": (res["net_profit"] / years / dd) if dd > 0 else np.nan,
        "sortino": sortino,
        "trades": res["n_trades"],
        "trades_per_year": res["n_trades"] / years,
        "win_rate": (wins / len(tr) * 100) if len(tr) else np.nan,
        "stops": res["n_stops"],
        "stops_per_year": res["n_stops"] / years,
        "spread_paid": res["spread_paid"],
        "swap_paid": res["swap_paid"],
    }
