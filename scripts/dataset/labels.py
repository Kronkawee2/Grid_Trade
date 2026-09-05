"""
What the model is asked to predict.

Not direction. Predicting whether price goes up or down is the hardest
problem in the field and the grid does not need it -- a grid makes money
when price oscillates inside a band and loses when price leaves the band
and keeps going. So the question worth asking is about *character*, not
direction: over the next H bars, will this market oscillate or travel?

The measure used is the efficiency ratio:

    ER = |close_last - close_first| / sum(|close_t - close_{t-1}|)

Net displacement over total distance travelled. A market that ends where
it started after moving a great deal scores near 0 -- every level was
revisited, which is exactly the condition a grid needs. A market that
moves in one direction without retracing scores 1, and a grid held
through it accumulates losing positions on one side with nothing closing.

ER is bounded in [0, 1] regardless of volatility or price level, so the
same threshold means the same thing on gold at 4,000 and on silver at 60,
and in a calm year as in a violent one. That property is why it is
preferred here over the obvious alternatives -- realised range, ATR
ratios, or a regression slope -- all of which need re-scaling per
instrument and per era.

The label is deliberately computed from *future* bars, which makes it
lookahead by construction. That is fine for a target and fatal for a
feature; the split module's embargo exists to stop the two from meeting.
"""

import numpy as np
import pandas as pd

# Bars ahead the label looks. Roughly a trading day at each timeframe,
# which is the horizon a grid position is expected to resolve within.
HORIZON = {"m5": 288, "m15": 96, "h1": 24, "h4": 6}

# ER at or below this counts as grid-friendly. 0.3 means the market
# travelled at least three times as far as it ended up moving.
#
# Not tuned -- picked as a round number before seeing any result, so that
# it doesn't quietly become a fitted parameter. Whether it lands in a
# sensible place is checked by looking at the class balance it produces.
CHOPPY_THRESHOLD = 0.30


def efficiency_ratio(close: pd.Series, horizon: int) -> pd.Series:
    """
    Forward-looking ER over the next `horizon` bars, per bar.

    The last `horizon` bars get NaN -- there is no future to measure yet,
    and filling them with anything would invent data.
    """
    step = close.diff().abs()
    # Distance travelled over the window ahead, and net displacement
    # across the same window.
    travelled = step.shift(-horizon).rolling(horizon).sum().shift(horizon - 1)
    travelled = step[::-1].rolling(horizon).sum()[::-1].shift(-1)
    displacement = (close.shift(-horizon) - close).abs()

    er = displacement / travelled.replace(0, np.nan)
    er[travelled == 0] = 0.0  # a flat window is maximally choppy
    return er.clip(0, 1)


def make_labels(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Adds `er` (continuous) and `grid_ok` (binary) to a bar frame.

    Both are returned because they answer different questions: `grid_ok`
    is what a classifier trains on, `er` is what tells you whether a
    borderline case was borderline.
    """
    timeframe = timeframe.lower()
    if timeframe not in HORIZON:
        raise KeyError(f"No label horizon defined for {timeframe!r} -- add one to HORIZON")

    out = df.copy()
    out["er"] = efficiency_ratio(out["close"], HORIZON[timeframe])
    out["grid_ok"] = (out["er"] <= CHOPPY_THRESHOLD).astype("float")
    out.loc[out["er"].isna(), "grid_ok"] = np.nan
    return out
