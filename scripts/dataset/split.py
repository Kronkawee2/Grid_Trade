"""
The train / out-of-sample boundary. Fixed here, in one place, on purpose.

The whole value of a held-out period is that it was chosen before anyone
saw a result on it. A cutoff that gets nudged after a disappointing score
is not out-of-sample any more, it is a slower kind of overfitting. So the
dates live in this module as constants, and every script imports them
rather than slicing by hand.

If a cutoff genuinely has to move, change it here, say why in the commit,
and re-run everything -- do not special-case it at a call site.

Why the dates differ per timeframe
----------------------------------
A single shared cutoff would have been cleaner, since the models would
then be judged on exactly the same market conditions. It isn't possible
here: the broker holds only 1.4 years of M5 (from April 2025), so a
cutoff late enough to leave M5 a usable training window would leave H1
almost nothing held out, and a cutoff early enough for H1 would leave M5
with a few weeks to train on.

The compromise is a shared cutoff for H1 and M15, which have the depth
for it, and a later one for M5. M5 results are therefore evaluated on a
different, shorter period than the others -- they are not directly
comparable, and M5's four-month window is thin enough that a good score
there means considerably less than the same score on H1.
"""

import pandas as pd

# Shared cutoff for the timeframes deep enough to support it. One year of
# held-out data, ending at the edge of available history.
OOS_START = {
    "h1": pd.Timestamp("2025-09-01"),   # ~7.9y train / 1.0y OOS
    "m15": pd.Timestamp("2025-09-01"),  # ~3.2y train / 1.0y OOS
    "m5": pd.Timestamp("2026-05-01"),   # ~1.0y train / 0.4y OOS
}

# A gap between the end of training and the start of the held-out period,
# so that a label computed from future bars at the end of training cannot
# overlap bars the model is later evaluated on. Without it the two sets
# share information and OOS scores come out flattering.
EMBARGO = {
    "h1": pd.Timedelta(days=7),
    "m15": pd.Timedelta(days=2),
    "m5": pd.Timedelta(days=1),
}


def split(df: pd.DataFrame, timeframe: str):
    """
    Split a bar-indexed frame into (train, oos).

    Bars inside the embargo window belong to neither and are dropped.
    """
    timeframe = timeframe.lower()
    if timeframe not in OOS_START:
        raise KeyError(
            f"No split defined for {timeframe!r}. Add one to OOS_START and "
            f"EMBARGO in scripts/dataset/split.py -- deliberately not "
            f"defaulted, so a new timeframe forces the decision to be made."
        )
    cutoff = OOS_START[timeframe]
    train_end = cutoff - EMBARGO[timeframe]
    return df.loc[df.index < train_end], df.loc[df.index >= cutoff]


def describe(df: pd.DataFrame, timeframe: str) -> str:
    train, oos = split(df, timeframe)
    def span(part):
        if part.empty:
            return "empty"
        years = (part.index[-1] - part.index[0]).days / 365.25
        return f"{len(part):,} bars, {part.index[0]:%Y-%m-%d} -> {part.index[-1]:%Y-%m-%d} ({years:.2f}y)"
    return (f"{timeframe}: train {span(train)} | oos {span(oos)}")
