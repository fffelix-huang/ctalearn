import numpy as np
import polars as pl
from numba import njit
from numpy.typing import NDArray

from ctalearn.core.dataframe import DataFrame
from ctalearn.operator._utils import numba_rolling_std


def ts_rank(df: DataFrame, window: int, constant: float = 0.0) -> DataFrame:
    """
    Rank the values for each column over the past window size.

    Returns the rank of the current value plus constant.
    Rank is scaled to [0, 1] range.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.
    constant : float, default=0.0
        Constant to add to the rank.

    Returns
    -------
    DataFrame
        DataFrame with ranked values.

    Raises
    ------
    TypeError
        If `df` is not of type DataFrame.
    ValueError
        If `window` is not a positive integer.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_rank({df.name}, {window}, constant={constant})"

    # Edge case: window == 1 -> rank of a lone value is the midpoint 0.5.
    # Still honor `constant` and null out NaN/inf, like the window>=2 path.
    if window == 1:
        return df.select(
            pl.col(df.time_col),
            (pl.exclude(df.time_col) * 0 + 0.5 + constant).fill_nan(None),
        )

    @njit(  # type: ignore[untyped-decorator]
        cache=True,
        nogil=True,
        parallel=False,
        fastmath={"nsz", "ninf", "reassoc", "arcp", "contract", "afn"},
    )
    def _numba_rolling_rank(
        arr: NDArray[np.float64], window: int, constant: float
    ) -> NDArray[np.float64]:
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)

        if n < window:
            return result

        denom = 1.0 / (window - 1)
        raw_buffer = np.full(window, np.nan, dtype=np.float64)
        nan_count = window

        for i in range(n):
            current_val = arr[i]
            idx = i % window

            if np.isnan(raw_buffer[idx]):
                nan_count -= 1

            raw_buffer[idx] = current_val

            if np.isnan(raw_buffer[idx]):
                nan_count += 1

            if nan_count > 0:
                continue

            less_count = 0
            equal_count = 0

            for x in raw_buffer:
                if x < current_val:
                    less_count += 1
                elif x == current_val:
                    equal_count += 1

            rank = less_count + (equal_count - 1) * 0.5
            result[i] = (rank * denom) + constant

        return result

    target_cols = [c for c in df._df.columns if c != df.time_col]

    return df.select(
        pl.col(df.time_col),
        *[
            pl.col(col)
            .map_batches(
                lambda s: _numba_rolling_rank(
                    s.to_numpy().astype(np.float64), window, constant
                ),
                return_dtype=pl.Float64,
            )
            .fill_nan(None)
            .alias(col)
            for col in target_cols
        ],
    )


def ts_mean(df: DataFrame, window: int) -> DataFrame:
    """
    Returns average value over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling mean values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_mean({df.name}, {window})"

    return df.select(
        pl.col(df.time_col), pl.exclude(df.time_col).rolling_mean(window_size=window)
    )


def ts_median(df: DataFrame, window: int) -> DataFrame:
    """
    Returns median value over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling median values. If window is even, return the
        average of the 2 center values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_median({df.name}, {window})"

    return df.select(
        pl.col(df.time_col), pl.exclude(df.time_col).rolling_median(window_size=window)
    )


def ts_std_dev(df: DataFrame, window: int, ddof: int = 0) -> DataFrame:
    """
    Return standard deviation over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.
    ddof : int
        Delta degree of freedom.

    Returns
    -------
    DataFrame
        DataFrame with rolling standard deviation values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_std_dev({df.name}, {window}, ddof={ddof})"

    return df.select(
        pl.col(df.time_col),
        pl.exclude(df.time_col).rolling_std(window_size=window, ddof=ddof),
    )


def ts_zscore(df: DataFrame, window: int) -> DataFrame:
    """
    Return z-score over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling z-score values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_zscore({df.name}, {window})"

    target = pl.exclude(df.time_col)

    zscore_expr = (
        (target - target.rolling_mean(window))
        / target.rolling_std(window, ddof=0).fill_null(0)
    ).fill_nan(0)

    return df.select(pl.col(df.time_col), zscore_expr)


def ts_robust_zscore(df: DataFrame, window: int) -> DataFrame:
    """
    Calculate robust Z-score using median and median absolute deviation (MAD).

    Formula: (x - median) / (c * MAD)
    Constant c = 1.4826.
    Returns 0 if MAD is 0 (constant window).
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    final_name = f"ts_robust_zscore({df.name}, {window})"

    @njit(  # type: ignore[untyped-decorator]
        cache=True,
        nogil=True,
        parallel=False,
        fastmath={"nsz", "ninf", "reassoc", "arcp", "contract", "afn"},
    )
    def _numba_robust_zscore(
        arr: NDArray[np.float64], medians: NDArray[np.float64], window: int
    ) -> NDArray[np.float64]:
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)

        if n < window:
            return result

        mad_scale = 1.4826
        raw_buffer = np.full(window, np.nan, dtype=np.float64)
        nan_count = window

        for i in range(n):
            idx = i % window

            if np.isnan(raw_buffer[idx]):
                nan_count -= 1

            raw_buffer[idx] = arr[i]

            if np.isnan(raw_buffer[idx]):
                nan_count += 1

            if nan_count > 0:
                continue

            median = medians[i]
            if np.isnan(median):
                continue

            mad = np.median(np.abs(raw_buffer - median))

            adjusted_mad = mad_scale * mad
            if adjusted_mad > 1e-12:
                result[i] = (arr[i] - median) / adjusted_mad
            else:
                result[i] = 0.0

        return result

    target_cols = [c for c in df._df.columns if c != df.time_col]

    df_median = ts_median(df, window=window).rename(
        {col: f"{col}_temp" for col in target_cols}
    )
    df = df.concat(df_median)

    df = df.select(
        pl.col(df.time_col),
        *[
            pl.struct(
                [
                    pl.col(col),
                    pl.col(f"{col}_temp"),
                ]
            )
            .map_batches(
                lambda s, col=col: _numba_robust_zscore(  # type: ignore[misc]
                    s.struct.field(col).to_numpy().astype(np.float64),
                    s.struct.field(f"{col}_temp").to_numpy().astype(np.float64),
                    window,
                ),
                return_dtype=pl.Float64,
            )
            .fill_nan(None)
            .alias(col)
            for col in target_cols
        ],
    )

    df.name = final_name
    return df


def ts_sum(df: DataFrame, window: int) -> DataFrame:
    """
    Returns sum of values over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling sum values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_sum({df.name}, {window})"

    return df.select(
        pl.col(df.time_col), pl.exclude(df.time_col).rolling_sum(window_size=window)
    )


def ts_min(df: DataFrame, window: int) -> DataFrame:
    """
    Returns min of values over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling min values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_min({df.name}, {window})"

    return df.select(
        pl.col(df.time_col), pl.exclude(df.time_col).rolling_min(window_size=window)
    )


def ts_max(df: DataFrame, window: int) -> DataFrame:
    """
    Returns max of values over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling max values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_max({df.name}, {window})"

    return df.select(
        pl.col(df.time_col), pl.exclude(df.time_col).rolling_max(window_size=window)
    )


def ts_scale(df: DataFrame, window: int, constant: float = 0.0) -> DataFrame:
    """
    Returns min-max scaling over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling max values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    final_name = f"ts_scale({df.name}, {window}, constant={constant})"

    mins = ts_min(df, window=window)
    maxs = ts_max(df, window=window)
    result_df = (df - mins) / (maxs - mins + 1e-12) + constant
    result_df.name = final_name
    return result_df


def ts_decay_linear(df: DataFrame, window: int, dense: bool = True) -> DataFrame:
    """
    Returns the linear decay on values over the past `window` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    window : int
        Size of the rolling window.
    dense : bool, default=True
        If True, returns NaN if any value in window is NaN.
        If False, treat NaN as 0.

    Returns
    -------
    DataFrame
        DataFrame with linearly decayed values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df = df.copy()
    df.name = f"ts_decay_linear({df.name}, {window}, dense={dense})"

    weights = [float(w) for w in range(1, window + 1)]

    def _apply_decay(col_name: str) -> pl.Expr:
        c = pl.col(col_name)
        values = (
            c.fill_null(0).fill_nan(0).rolling_mean(weights=weights, window_size=window)
        )

        if dense:
            has_invalid = (c.is_null() | c.is_nan()).rolling_max(window_size=window)
            mask = pl.when(has_invalid).then(None).otherwise(1.0)
            return (values * mask).alias(col_name)
        else:
            return values.alias(col_name)

    target_cols = [c for c in df._df.columns if c != df.time_col]

    return df.select(pl.col(df.time_col), *[_apply_decay(c) for c in target_cols])


def ts_delay(df: DataFrame, d: int) -> DataFrame:
    """
    Returns values delayed by `d` periods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    d : int
        Lag periods.

    Returns
    -------
    DataFrame
        DataFrame with lagged values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if d <= 0:
        raise ValueError(f"`d` must be a positive integer, got {d} instead")

    df = df.copy()
    df.name = f"ts_delay({df.name}, {d})"

    return df.select(pl.col(df.time_col), pl.exclude(df.time_col).shift(d))


def ts_delta(df: DataFrame, d: int) -> DataFrame:
    """
    Returns x - ts_delay(x, d).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    d : int
        Lag periods.

    Returns
    -------
    DataFrame
        DataFrame with delta values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if d <= 0:
        raise ValueError(f"`d` must be a positive integer, got {d} instead")

    df = df.copy()
    df.name = f"ts_delta({df.name}, {d})"

    return df.select(pl.col(df.time_col), pl.exclude(df.time_col).diff(d))


def ts_ffill(df: DataFrame, limit: int | None = None) -> DataFrame:
    """
    Perform forward fill.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    limit : int | None
        Max consecutive values to fill.

    Returns
    -------
    DataFrame
        Forward-filled DataFrame.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if limit is not None and limit <= 0:
        raise ValueError(f"`limit` must be a positive integer, got {limit} instead")

    df = df.copy()
    df.name = f"ts_ffill({df.name}, limit={limit})"

    return df.select(
        pl.col(df.time_col),
        pl.exclude(df.time_col).fill_null(strategy="forward", limit=limit),
    )


def ts_corr(df_a: DataFrame, df_b: DataFrame, window: int) -> DataFrame:
    """
    Compute rolling correlation between common columns of two DataFrames.

    Performs an inner join on their respective `time_col` to align timestamps,
    then calculates the rolling correlation for columns that exist in both DataFrames.

    Parameters
    ----------
    df_a : DataFrame
        First input DataFrame.
    df_b : DataFrame
        Second input DataFrame.
    window : int
        Size of the rolling window.

    Returns
    -------
    DataFrame
        DataFrame with rolling correlation values for common columns.
        The time column name will follow `df_a.time_col`.
    """
    if not isinstance(df_a, DataFrame):
        raise TypeError(
            f"Type of `df_a` should be DataFrame, got {type(df_a).__name__}"
        )
    if not isinstance(df_b, DataFrame):
        raise TypeError(
            f"Type of `df_b` should be DataFrame, got {type(df_b).__name__}"
        )

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    df_a = df_a.copy()
    df_b = df_b.copy()

    time_a = df_a.time_col
    time_b = df_b.time_col

    cols_a = set(df_a._df.columns) - {time_a}
    cols_b = set(df_b._df.columns) - {time_b}
    common_cols = sorted(list(cols_a & cols_b))

    if not common_cols:
        return DataFrame(
            df_a._df.select(pl.col(time_a)).clear(),
            time_a,
            df_a.freq,
            "None",
            _skip_validate=True,
        )

    if time_a == time_b:
        joined = df_a._df.join(df_b._df, on=time_a, how="inner", suffix="_right")
    else:
        joined = df_a._df.join(
            df_b._df, left_on=time_a, right_on=time_b, how="inner", suffix="_right"
        )

    select_exprs = [pl.col(time_a)]

    for col in common_cols:
        expr = pl.rolling_corr(
            pl.col(col), pl.col(f"{col}_right"), window_size=window
        ).alias(col)
        select_exprs.append(expr)

    result_pl = joined.select(select_exprs)

    return DataFrame(
        result_pl, time_a, df_a.freq, f"ts_corr({df_a.name}, {df_b.name}, {window})"
    )


@njit(cache=True)  # type: ignore[untyped-decorator]
def _hurst_slopes(
    arr: NDArray[np.float64],
    window: int,
    lag_start: int,
    lag_end: int,
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rolling Hurst slope via a closed-form degree-1 OLS fit.

    The slope of log10(tau) on the fixed regressor log10(lag) is `weights . y`,
    where `weights = (x - mean(x)) / sum((x - mean(x))**2)` is precomputed once
    by the caller. This avoids a per-timestamp polyfit/SVD: each lag contributes
    a single streaming rolling-std pass whose log is accumulated into the slope.
    A timestamp is emitted only when every lag produced a finite, positive std.
    """
    n = arr.shape[0]
    result = np.full(n, np.nan)
    if n < window:
        return result

    num_lags = lag_end - lag_start
    acc = np.zeros(n)
    cnt = np.zeros(n, dtype=np.int64)

    for li in range(num_lags):
        lag = lag_start + li
        if window - lag < 2:
            # A lag with no usable window can never produce a slope, so no row
            # is ever valid -> the whole column is null (matches the reference).
            return result
        diff = arr[lag:] - arr[:-lag]
        rstd = numba_rolling_std(diff, window - lag, 0)
        w = weights[li]
        for j in range(rstd.shape[0]):
            v = rstd[j]
            if v > 0.0 and not np.isnan(v):
                t = lag + j
                acc[t] += w * np.log10(v)
                cnt[t] += 1

    for t in range(window - 1, n):
        if cnt[t] == num_lags:
            result[t] = acc[t]
    return result


def ts_hurst_exponent(
    df: DataFrame, window: int, lag_start: int = 2, lag_end: int = 20
) -> DataFrame:
    """
    Rolling Hurst exponent estimate (slope of log-log plot of lag vs std of differences).

    For each rolling window, compute:
      tau(lag) = std(x[t] - x[t-lag])
      hurst = slope of log10(tau) regressed on log10(lag)

    Notes
    -----
    - The degree-1 fit is solved in closed form (`weights . y`) inside a single
      Numba kernel; mathematically identical to a least-squares polyfit, agreeing
      to ~1e-11 relative.
    - If a window contains NaN/null or has insufficient valid lags, the output is null for that row.
    - `lag_start`/`lag_end` follow Python `range(start, end)` semantics (end is exclusive),
      matching the common reference implementation.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 0:
        raise ValueError(f"`window` must be a positive integer, got {window} instead")

    lag_start, lag_end = int(lag_start), int(lag_end)
    lag_end = min(lag_end, window)

    if lag_start <= 0 or lag_end <= 0 or lag_end < lag_start:
        raise ValueError(
            f"`lag_start`/`lag_end` must satisfy 0 < start <= end, "
            f"got ({lag_start}, {lag_end})"
        )
    if lag_end - lag_start < 2:
        raise ValueError(f"lag_end - lag_start < 2, all values will be null.")

    df = df.copy()
    df.name = f"ts_hurst_exponent({df.name}, {window}, {lag_start}, {lag_end})"

    # OLS slope weights for the fixed regressor log10(lag): slope = weights . y.
    x = np.log10(np.arange(lag_start, lag_end, dtype=np.float64))
    xc = x - x.mean()
    weights = xc / (xc @ xc)

    target_cols = [col for col in df._df.columns if col != df.time_col]

    return df.select(
        pl.col(df.time_col),
        *[
            pl.col(col)
            .map_batches(
                lambda s: _hurst_slopes(
                    s.to_numpy().astype(np.float64),
                    window,
                    lag_start,
                    lag_end,
                    weights,
                ),
                return_dtype=pl.Float64,
            )
            .fill_nan(None)
            .alias(col)
            for col in target_cols
        ],
    )
