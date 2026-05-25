import numpy as np
import polars as pl

from ctalearn.core.dataframe import DataFrame


def trend(df: DataFrame, threshold: float) -> DataFrame:
    """
    Calculate trend positions based on threshold values.

    Logic:
    - 1  if x >= threshold
    - -1 if x <= -threshold
    - 0  if x is NaN (Input NaN handling)
    - Hold (Forward Fill) otherwise
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if threshold <= 0:
        raise ValueError(f"threshold must be greater than 0, got {threshold} instead")

    df = df.copy()
    df.name = f"trend({df.name}, {threshold})"

    target = pl.exclude(df.time_col)

    # Polars expression equivalent to the numpy logic:
    # Priority 1: Input is Null -> Signal 0
    # Priority 2: x >= threshold -> Signal 1
    # Priority 3: x <= -threshold -> Signal -1
    # Priority 4: Otherwise -> Null (which represents "Hold")
    # Finally: Forward Fill to propagate signals during "Hold" periods

    return df.select(
        pl.col(df.time_col),
        pl.when(target.is_null() | target.is_nan())
        .then(0.0)
        .when(target >= threshold)
        .then(1.0)
        .when(target <= -threshold)
        .then(-1.0)
        .otherwise(None)
        .fill_null(strategy="forward")
        .name.keep(),
    )


def trend_reverse(df: DataFrame, threshold: float) -> DataFrame:
    """
    Calculate trend reverse positions.
    Simple negation of trend strategy.
    """
    return -trend(df, threshold)


def trend_mean_reversion(df: DataFrame, threshold: float) -> DataFrame:
    """
    Calculate mean reversion positions.

    Logic:
    - 1 (Long) if x >= threshold
    - -1 (Short) if x <= -threshold
    - 0 (Close) if x crosses 0 (positive or negative crossover) AND not opening a position
    - 0 if x is NaN
    - Hold (Forward Fill) otherwise
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if threshold <= 0:
        raise ValueError(f"threshold must be greater than 0, got {threshold} instead")

    df = df.copy()
    df.name = f"trend_mean_reversion({df.name}, {threshold})"

    # Define expressions for clarity
    curr = pl.exclude(df.time_col)

    # Prev value, fill initial null/NaN with 0 to match numpy logic (x_shifted[0, :] = 0)
    prev = curr.shift(1).fill_null(0.0).fill_nan(0.0)

    # 1. Crossing Logic
    # Cross positive: Prev < 0 AND Curr > 0
    cross_pos = (prev < 0) & (curr > 0)
    # Cross negative: Prev > 0 AND Curr < 0
    cross_neg = (prev > 0) & (curr < 0)

    cross_zero = cross_pos | cross_neg

    # 2. Construct Signal Logic
    # Note: The order of `when` clauses determines priority.
    # In original numpy code: open triggers override close triggers.
    # signals[close_triggers] = 0 where close_triggers = (cross) & (~open)
    # This implies Open Logic has precedence over Close Logic.

    return df.select(
        pl.col(df.time_col),
        pl.when(curr.is_null() | curr.is_nan())
        .then(0.0)  # Input NaN -> 0
        .when(curr >= threshold)
        .then(1.0)  # Open Long
        .when(curr <= -threshold)
        .then(-1.0)  # Open Short
        .when(cross_zero)
        .then(0.0)  # Close Position (Cross 0)
        .otherwise(None)  # Hold
        .fill_null(strategy="forward")
        .name.keep(),
    )


def trend_reverse_mean_reversion(df: DataFrame, threshold: float) -> DataFrame:
    """
    Calculate reverse mean reversion positions.
    Negation of trend_mean_reversion.
    """
    return -trend_mean_reversion(df, threshold)


def trend_fast(df: DataFrame, threshold: float) -> DataFrame:
    """
    Calculate fast trend positions.
    Logic:
    - 1 (Long) if x >= threshold
    - -1 (Short) if x <= -threshold
    - 0 (Close) if x crosses threshold (x < threshold or x > -threshold) AND not opening a position
    - 0 if x is NaN
    - Hold (Forward Fill) otherwise
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if threshold <= 0:
        raise ValueError(f"threshold must be greater than 0, got {threshold} instead")

    df = df.copy()
    df.name = f"trend_fast({df.name}, {threshold})"

    curr = pl.exclude(df.time_col)
    prev = curr.shift(1).fill_null(0.0).fill_nan(0.0)
    cross_threshold = (prev >= threshold) & (curr < threshold) | (
        prev <= -threshold
    ) & (curr > -threshold)

    return df.select(
        pl.col(df.time_col),
        pl.when(curr.is_null() | curr.is_nan())
        .then(0.0)
        .when(curr >= threshold)
        .then(1.0)
        .when(curr <= -threshold)
        .then(-1.0)
        .when(cross_threshold)
        .then(0.0)
        .otherwise(None)
        .fill_null(strategy="forward")
        .name.keep(),
    )


def trend_reverse_fast(df: DataFrame, threshold: float) -> DataFrame:
    """
    Calculate reverse fast trend positions.
    Negation of trend_fast.
    """
    return -trend_fast(df, threshold)


def trend_time(df: DataFrame, threshold: float, step: int) -> DataFrame:
    """Fixed-duration trend signal.

    Logic (per column):
    - Start long (1) when x >= threshold
    - Start short (-1) when x <= -threshold
    - Once started, keep the position for `step` rows (including the trigger row)
    - After `step` rows, revert back to 0
    - Null values do not create triggers; output is determined by the most recent trigger
      within the last `step` rows (so the window can continue through nulls)

    Notes
    -----
    This is a "pulse"-style signal (duration-limited), not a forward-filled hold-until-close.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if threshold <= 0:
        raise ValueError(f"threshold must be greater than 0, got {threshold} instead")

    if step <= 0:
        raise ValueError(f"step must be greater than 0, got {step} instead")

    df = df.copy()
    df.name = f"trend_time({df.name}, {threshold}, step={step})"

    time_col = df.time_col
    value_cols = [c for c in df.columns if c != time_col]
    idx = pl.int_range(0, pl.len())

    exprs: list[pl.Expr] = []
    for c in value_cols:
        x = pl.col(c)
        # NaN/null must not trigger (note: NaN >= threshold is True in Polars)
        valid = x.is_not_null() & x.is_not_nan()
        trigger = (
            pl.when(valid & (x >= threshold))
            .then(1.0)
            .when(valid & (x <= -threshold))
            .then(-1.0)
            .otherwise(None)
        )

        # O(n) per column:
        # 1) Forward-fill last trigger value and its row index
        # 2) Keep it only if it's within `step` rows; otherwise output 0
        last_val = trigger.fill_null(strategy="forward")
        last_idx = (
            pl.when(trigger.is_not_null())
            .then(idx)
            .otherwise(None)
            .fill_null(strategy="forward")
        )

        exprs.append(
            pl.when(last_idx.is_null())
            .then(0.0)
            .when((idx - last_idx) < step)
            .then(last_val)
            .otherwise(0.0)
            .alias(c)
        )

    return df.select(pl.col(time_col), *exprs)
