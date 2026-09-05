"""
Adaptive grid: spacing scales with volatility, exposure is capped, and the
grid steps aside when the market is moving too fast to fade.

Why this replaces the linear grid
---------------------------------
The fixed grid was measured over nineteen years and the verdict was not
that it made too little money -- it was that its losses were unbounded.
Walk-forward chose configurations whose training drawdown was $150 and
whose live drawdown was $2,100, because a deep grid hides losses inside
open positions and a training window that happens to end in a recovery
never shows the bill.

Three changes, each aimed at that:

1. Spacing follows ATR instead of being a constant. A 42-pip grid is
   loose in 2019 and suicidal in 2008; the same number cannot be right in
   both. Spacing is frozen when a cycle anchors, so a cycle's geometry
   never shifts underneath its own open positions.

2. Concurrent positions are capped. This is the change that actually
   bounds risk, and it does so structurally rather than statistically:
   with `max_open` positions and a stop `stop_mult` steps out, the worst
   a cycle can cost is known before it starts, in every market that has
   ever existed or ever will. Nothing about the historical sample is
   being trusted.

3. Entries stop while short-term volatility runs far above its own
   baseline. A grid earns by fading noise, and a market moving three
   times its normal speed is not producing noise -- it is producing the
   one-directional run that grids die in.

The cap is what makes the other two useful. Without it, adaptive spacing
just resizes an unbounded loss.
"""

import numpy as np
import pandas as pd


# A break longer than a normal weekend means the bars either side are not
# consecutive observations of the same market.
MAX_CONTINUOUS_BREAK = pd.Timedelta("3D")


def true_range(df: pd.DataFrame) -> np.ndarray:
    """
    True range, with the previous close dropped across a discontinuity.

    The usual definition reaches back one bar for a closing price. Where
    history is missing that neighbour can be days away, and the difference
    between the two is booked as a single bar's range: on the gold feed,
    with 466 days lost to rate limiting, the bar after a hole showed a true
    range ten times normal and in one case a hundred times.

    That number does not stay local. It feeds a 200-bar average, which sets
    the grid spacing, which sets the stop -- so a download failure in 2011
    silently widens the grid for the next seventeen hours of trading and
    nothing downstream can see why. Where there is no trustworthy previous
    close, the bar's own high-low is the honest answer.
    """
    prev_close = df["close"].shift(1)
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(dtype=float),
        (df["high"] - prev_close).abs().to_numpy(dtype=float),
        (df["low"] - prev_close).abs().to_numpy(dtype=float),
    ])
    broken = df.index.to_series().diff() > MAX_CONTINUOUS_BREAK
    if broken.any():
        own = (df["high"] - df["low"]).to_numpy(dtype=float)
        tr = np.where(broken.to_numpy(), own, tr)
    return tr


def efficiency_ratio(df: pd.DataFrame, window: int) -> np.ndarray:
    """
    Kaufman's efficiency ratio: net distance travelled over total distance
    travelled, across `window` bars. Near 1 the market went somewhere in a
    straight line; near 0 it went nowhere the long way round.

    This is the measurement a grid actually needs, and volatility is not.
    A fast market that keeps returning to the same place is a grid's best
    friend; a slow market that never comes back is what kills it, and the
    two can have identical ATR. Blocking on ATR alone therefore stands
    aside during exactly the volatile chop that pays, while walking into
    quiet one-way drifts.
    """
    c = df["close"].to_numpy(dtype=float)
    move = np.abs(c[window:] - c[:-window])
    step = np.abs(np.diff(c))
    total = pd.Series(step).rolling(window).sum().to_numpy()[window - 1:]
    er = np.full(len(c), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        er[window:] = np.where(total > 0, move / total, 0.0)
    return pd.Series(er).ffill().fillna(0.0).to_numpy()


def volatility(df: pd.DataFrame, fast: int, slow: int, pip: float):
    """
    Return (atr_pips, ratio). `ratio` is the fast ATR over the slow one --
    a unitless measure of "faster than usual", which transfers across
    instruments and across decades in a way a raw ATR never does. It is
    the same construction as the reference system's atr_window against
    atr_baseline_window.

    Bars where the market is shut carry a zero range and would drag both
    averages toward zero, so they are excluded from the averaging rather
    than counted as calm. On this dataset that is 17% of all bars, every
    one of them a Sunday.
    """
    tr = pd.Series(true_range(df), index=df.index)
    live = df["high"].to_numpy() > df["low"].to_numpy()
    tr = tr.where(live)
    atr = tr.rolling(fast, min_periods=max(2, fast // 2)).mean()
    base = tr.rolling(slow, min_periods=max(5, slow // 4)).mean()
    atr_pips = (atr / pip).ffill().bfill().to_numpy(dtype=float)
    ratio = (atr / base).ffill().bfill().to_numpy(dtype=float)
    return atr_pips, ratio


class AdaptiveConfig:
    def __init__(self, atr_mult=4.0, max_open=3, stop_extra=2.0,
                 atr_fast=20, atr_slow=200, vol_block=1.8,
                 er_window=0, er_block=1.0, dd_pause=None, dd_pause_bars=0,
                 rebound=0.0, vol_floor=0.0, equity_stop=None,
                 equity_stop_pct=None, compound=False,
                 equity_per_lot_step=100.0, start_cash=100.0, lot_step=0.01,
                 max_lot=None, scale_stop_with_lot=True,
                 lot=0.01, pip=0.0001, pip_value_per_lot=100_000 * 0.0001,
                 spread_pips=1.0, swap_long_pips=-0.7, swap_short_pips=0.25,
                 side="both", max_hold_days=None):
        # Grid step, in ATRs. Below ~2 the step is inside the noise the
        # spread already eats; above ~8 the grid rarely fills at all.
        self.atr_mult = atr_mult
        # The risk cap. Everything else is optimisation; this is the part
        # that decides whether the account survives.
        self.max_open = max_open
        # Steps BEYOND the deepest level, not from the anchor. Measured
        # from the anchor it is trivially possible to place the stop on
        # top of the last entry -- stop_mult 3 with three levels did
        # exactly that, and the bar that filled level three stopped it in
        # the same instant. Expressed this way the mistake cannot be made.
        self.stop_extra = stop_extra
        self.atr_fast = atr_fast
        self.atr_slow = atr_slow
        # Stand aside above this fast/slow ATR ratio. 1.0 is "normal".
        self.vol_block = vol_block
        # ...and below this one. Measured on nineteen years, the quiet
        # third of the distribution was the only losing regime for a wide
        # grid: spacing is a multiple of ATR, so a dead market shrinks the
        # take-profit toward the spread while the stop stays a fixed
        # number of steps away. High volatility, by contrast, widens the
        # grid with it and turned out to be the most profitable bucket --
        # the opposite of the usual advice, and the reason this floor
        # exists as a searchable number rather than a fixed rule.
        self.vol_floor = vol_floor
        # Trend gate. er_window=0 disables it; otherwise entries stop
        # while the efficiency ratio is above er_block, i.e. while price
        # is travelling in a line rather than oscillating.
        self.er_window = er_window
        self.er_block = er_block
        # Circuit breaker. The 2025 out-of-sample run reached its $108
        # drawdown before it earned anything, so a $100 account would have
        # been closed months before the profit arrived. Order matters as
        # much as total, and a breaker is the only thing that acts on
        # order: past this loss from the equity peak, stand down for a
        # while instead of continuing to average into whatever caused it.
        self.dd_pause = dd_pause
        self.dd_pause_bars = dd_pause_bars
        # Hard equity stop, in account currency below the equity peak.
        # Unlike dd_pause -- which only stops NEW entries and so lets the
        # positions already open keep running to their own stop -- this
        # closes everything at once. It is the only control here that
        # bounds drawdown by construction rather than by choosing
        # parameters and hoping.
        #
        # The cost is not small and should be expected: a grid's floating
        # losses are usually temporary, and this converts them into
        # realised ones at the worst moment, right before the recovery it
        # was waiting for. It buys a survivable account, not a better one.
        self.equity_stop = equity_stop
        # The same cap expressed as a share of live equity. Everything else
        # in this design is a multiple of ATR or a ratio, so a threshold
        # fixed in dollars was the one piece that did not travel: on an
        # account that grew from $100 to $800 the same $22 silently became
        # eight times tighter, and on silver -- which moves $0.50 a pip at
        # the broker's minimum lot against EURUSD's $0.10 -- it was too
        # small to catch anything before the damage was done.
        self.equity_stop_pct = equity_stop_pct
        # Fixed-fractional sizing. Position count stays capped at max_open
        # because that cap is what makes the worst case knowable in
        # advance; only the lot grows. Brokers quantise lots (0.01 here),
        # so on a small account this compounds in visible steps rather
        # than smoothly -- $100 to $200 before the size can change at all.
        self.compound = compound
        self.equity_per_lot_step = equity_per_lot_step
        self.start_cash = start_cash
        self.lot_step = lot_step
        # The broker will not fill above this, so neither may the backtest.
        # Without it a compounding run on gold reached 207 lots against a
        # 30-lot ceiling and reported turning $100 into two million.
        self.max_lot = max_lot
        # The cash equity stop has to grow with the position, or it stops
        # meaning anything. Held at $22 while the lot compounded to 207,
        # it came to 0.0106 pips -- a hundredth of a pip, against a spread
        # of 1.6. A threshold far inside the spread is not a risk control;
        # it is noise being read as a decision.
        self.scale_stop_with_lot = scale_stop_with_lot
        self.base_lot = lot
        self.pip_value_per_lot = pip_value_per_lot
        # Momentum confirmation, as a fraction of one grid step. At 0 a
        # level is a plain resting limit: price touches it and the
        # position exists, which is the classic way to catch a falling
        # knife. Above 0 the touch only arms the level, and the entry
        # waits until price has come back `rebound` steps in the trade's
        # favour -- evidence that the move paused, rather than a guess
        # that it will.
        #
        # This is the layer aimed straight at the break-even problem. The
        # grid needs a 74.9% win rate to pay for its own stop and delivers
        # 73-81% depending on the era; every skipped knife is a stop that
        # never happens.
        #
        # The cost is real: entries are worse by `rebound` steps, and
        # levels that never bounce are never traded at all.
        self.rebound = rebound
        self.lot = lot
        self.pip = pip
        # Account currency per pip at this lot size.
        self.pip_value = pip_value_per_lot * lot
        self.spread_pips = spread_pips
        self.swap_long = swap_long_pips
        self.swap_short = swap_short_pips
        self.side = side
        # Optional time stop. A grid's real losses are not deep, they are
        # long: the worst adverse excursion in this dataset took 240 days
        # to come back.
        self.max_hold_days = max_hold_days

    def stop_steps(self):
        return self.max_open + self.stop_extra

    def worst_case_per_cycle(self, spacing_pips):
        """
        What one cycle can lose, known before it is opened. Each of the
        max_open positions sits at most stop_steps from the anchor, so
        this is an upper bound rather than an estimate from the sample.
        """
        return self.max_open * self.stop_steps() * spacing_pips * self.pip_value


class Pos:
    __slots__ = ("side", "entry", "tp", "opened_at", "opened_day", "opened_i", "k", "lot")

    def __init__(self, side, entry, tp, opened_at, opened_day, opened_i, k, lot):
        self.side = side
        self.entry = entry
        self.tp = tp
        self.opened_at = opened_at
        self.opened_day = opened_day
        self.opened_i = opened_i
        self.k = k
        self.lot = lot


def run_adaptive(df: pd.DataFrame, cfg: AdaptiveConfig) -> dict:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    times = df.index.to_numpy()
    days = df.index.normalize().to_numpy()
    live = high > low                     # a bar with no range is a shut market

    atr_pips, vol_ratio = volatility(df, cfg.atr_fast, cfg.atr_slow, cfg.pip)
    er = (efficiency_ratio(df, cfg.er_window) if cfg.er_window
          else np.zeros(len(df)))

    do_buy = cfg.side in ("buy", "both")
    do_sell = cfg.side in ("sell", "both")

    anchor = close[0]
    spacing = max(atr_pips[0] * cfg.atr_mult, 5.0) * cfg.pip
    open_pos: list[Pos] = []
    held: set[tuple[str, int]] = set()
    # Levels touched but not yet confirmed, and the bar the touch happened
    # on. A touch cannot confirm itself on its own bar: the high and low of
    # one bar carry no order, so "price fell to the level then bounced"
    # and "price bounced then fell to the level" are the same data.
    armed: dict[tuple[str, int], int] = {}

    realised = 0.0
    equity = np.empty(len(df))
    trades, boxes = [], []
    n_stops = n_resets = n_blocked = n_paused = n_equity_stops = 0
    pause_until = -1
    peak_equity = 0.0
    peak_open = 0          # audited, not assumed: the cap is the whole point
    box_start = 0

    def close_pos(p, exit_price, t, reason):
        nonlocal realised
        d = 1.0 if p.side == "buy" else -1.0
        gross = (exit_price - p.entry) / cfg.pip * d
        nights = int((t - p.opened_day) / np.timedelta64(1, "D"))
        swap = nights * (cfg.swap_long if p.side == "buy" else cfg.swap_short)
        net = gross - cfg.spread_pips + swap
        cash = net * cfg.pip_value_per_lot * p.lot
        realised += cash
        trades.append({
            "side": p.side, "entry": p.entry, "exit": exit_price,
            "opened_at": p.opened_at, "closed_at": t, "reason": reason,
            "gross_pips": gross, "swap_pips": swap, "net_pips": net,
            "net_cash": cash, "level": p.k, "lot": p.lot,
        })

    def reanchor(i, price):
        """Close the current box, open a new one sized by current ATR."""
        nonlocal anchor, spacing, box_start
        boxes.append({"start": times[box_start], "end": times[i],
                      "anchor": anchor, "spacing": spacing,
                      "levels": cfg.max_open})
        anchor = price
        armed.clear()
        # A floor keeps the grid from collapsing to nothing in dead hours,
        # where ATR can fall below the spread and every level would be a
        # guaranteed loss.
        spacing = max(atr_pips[i] * cfg.atr_mult, 5.0) * cfg.pip
        held.clear()
        box_start = i

    for i in range(len(df)):
        t, day = times[i], days[i]
        hi, lo, cl = high[i], low[i], close[i]

        if not live[i]:                    # market shut: no fills, no marking
            equity[i] = equity[i - 1] if i else 0.0
            continue

        stop_dist = spacing * (cfg.max_open + cfg.stop_extra)

        # --- stop -----------------------------------------------------
        if open_pos and (lo <= anchor - stop_dist or hi >= anchor + stop_dist):
            px = anchor - stop_dist if lo <= anchor - stop_dist else anchor + stop_dist
            for p in open_pos:
                close_pos(p, px, t, "stop")
            open_pos.clear()
            n_stops += 1
            reanchor(i, cl)
            equity[i] = realised
            continue

        # --- time stop ------------------------------------------------
        if cfg.max_hold_days and open_pos:
            oldest = min(int((t - p.opened_day) / np.timedelta64(1, "D")) for p in open_pos)
            if oldest >= cfg.max_hold_days:
                for p in open_pos:
                    close_pos(p, cl, t, "time")
                open_pos.clear()
                reanchor(i, cl)
                equity[i] = realised
                continue

        # --- entries, capped and regime-gated -------------------------
        calm = (cfg.vol_floor <= vol_ratio[i] <= cfg.vol_block
                and (not cfg.er_window or er[i] <= cfg.er_block)
                and i >= pause_until)
        if not calm and len(open_pos) < cfg.max_open:
            n_blocked += 1
        if calm:
            # Size is decided when a position opens and then travels with
            # it, so a grid opened small is not repriced by later growth.
            cur_lot = cfg.lot
            if cfg.compound:
                eq_now = cfg.start_cash + equity[i - 1] if i else cfg.start_cash
                steps = int(eq_now / cfg.equity_per_lot_step)
                cur_lot = max(cfg.lot, steps * cfg.lot_step)
                if cfg.max_lot:
                    cur_lot = min(cur_lot, cfg.max_lot)
            back = cfg.rebound * spacing
            for k in range(1, cfg.max_open + 1):
                for sd in (("buy",) if do_buy else ()) + (("sell",) if do_sell else ()):
                    key = (sd, k)
                    if key in held:
                        continue
                    lvl = anchor - k * spacing if sd == "buy" else anchor + k * spacing
                    touched = lo <= lvl if sd == "buy" else hi >= lvl
                    if touched and key not in armed:
                        armed[key] = i
                    if key not in armed or len(open_pos) >= cfg.max_open:
                        continue
                    if back == 0.0:
                        # No confirmation asked for: the touch is the entry.
                        entry = lvl
                        ready = armed[key] == i or True
                    else:
                        entry = lvl + back if sd == "buy" else lvl - back
                        # Confirmation must land on a later bar than the touch.
                        ready = armed[key] < i and (
                            hi >= entry if sd == "buy" else lo <= entry)
                    if ready:
                        held.add(key)
                        armed.pop(key, None)
                        tp = entry + spacing if sd == "buy" else entry - spacing
                        open_pos.append(Pos(sd, entry, tp, t, day, i, k, cur_lot))

        # --- take profit (never on the bar that opened the position) ---
        still = []
        for p in open_pos:
            if p.opened_i < i and ((p.side == "buy" and hi >= p.tp)
                                   or (p.side == "sell" and lo <= p.tp)):
                close_pos(p, p.tp, t, "tp")
                held.discard((p.side, p.k))
            else:
                still.append(p)
        open_pos = still

        # --- re-anchor when flat and out of reach ---------------------
        if not open_pos and abs(cl - anchor) > spacing * cfg.max_open:
            reanchor(i, cl)
            n_resets += 1

        peak_open = max(peak_open, len(open_pos))
        floating = 0.0
        for p in open_pos:
            d = 1.0 if p.side == "buy" else -1.0
            floating += (((cl - p.entry) / cfg.pip * d - cfg.spread_pips)
                         * cfg.pip_value_per_lot * p.lot)
        equity[i] = realised + floating

        peak_equity = max(peak_equity, equity[i])
        drop = peak_equity - equity[i]

        limit = cfg.equity_stop
        if limit and cfg.scale_stop_with_lot and open_pos:
            # Express the cap in price terms: whatever distance $limit
            # bought at the base lot, keep buying that same distance.
            limit *= max(p.lot for p in open_pos) / cfg.base_lot
        if cfg.equity_stop_pct:
            limit = cfg.equity_stop_pct * max(cfg.start_cash + equity[i], cfg.start_cash * 0.1)
        # A position may not be closed on the bar that opened it, for the
        # same reason it may not take profit there: the entry sits at a
        # limit level and the exit would be that bar's close, and nothing
        # in the data says which came first. Left unguarded, 86-92% of
        # equity-stop exits landed on their own opening bar and booked a
        # POSITIVE average gross result -- a forced loss that made money,
        # which is the signature of reading the future.
        closable = [p for p in open_pos if p.opened_i < i]
        if limit and drop >= limit and closable:
            for p in closable:
                close_pos(p, cl, t, "equity_stop")
            open_pos = [p for p in open_pos if p.opened_i == i]
            n_equity_stops += 1
            if not open_pos:
                reanchor(i, cl)
            pause_until = i + cfg.dd_pause_bars
            floating = 0.0
            for p in open_pos:
                d2 = 1.0 if p.side == "buy" else -1.0
                floating += (((cl - p.entry) / cfg.pip * d2 - cfg.spread_pips)
                             * cfg.pip_value_per_lot * p.lot)
            equity[i] = realised + floating
        elif cfg.dd_pause and drop >= cfg.dd_pause and i >= pause_until:
            pause_until = i + cfg.dd_pause_bars
            n_paused += 1

    boxes.append({"start": times[box_start], "end": times[-1],
                  "anchor": anchor, "spacing": spacing, "levels": cfg.max_open})

    eq = pd.Series(equity, index=df.index)
    dd = eq - eq.cummax()
    tr = pd.DataFrame(trades)

    days_live = max(len(np.unique(days[live])), 1)
    return {
        "equity": eq,
        "drawdown": dd,
        "trades": tr,
        "boxes": pd.DataFrame(boxes),
        "atr_pips": pd.Series(atr_pips, index=df.index),
        "vol_ratio": pd.Series(vol_ratio, index=df.index),
        "net_profit": realised,
        "max_drawdown": float(dd.min()),
        "n_trades": len(tr),
        "n_stops": n_stops,
        "n_resets": n_resets,
        "n_blocked_bars": n_blocked,
        "n_pauses": n_paused,
        "n_equity_stops": n_equity_stops,
        "er": pd.Series(er, index=df.index),
        "trading_days": days_live,
        "per_day": realised / days_live,
        "max_concurrent": peak_open,
        "win_rate": float((tr["net_pips"] > 0).mean() * 100) if len(tr) else np.nan,
        "open_at_end": len(open_pos),
    }
