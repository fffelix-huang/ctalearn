import math
from collections.abc import Callable
from datetime import timedelta
from functools import reduce, wraps
from typing import Any

import polars as pl


def _return_empty_aligned(
    df_self: "DataFrame",
    df_other: "DataFrame",
    method: Callable[..., "DataFrame"],
    common_cols: list[str],
    freq: int | None,
) -> "DataFrame":
    """Helper to return empty result with correct schema."""
    time_col = df_self.time_col
    other_time_col = df_other.time_col
    cols_to_keep = [time_col] + common_cols

    df1_empty = df_self._df.clear().select([pl.col(c) for c in cols_to_keep])
    df2_empty = (
        df_other._df.clear()
        .rename({other_time_col: time_col})
        .select([pl.col(c) for c in cols_to_keep])
    )

    return method(
        df_self,
        DataFrame(df1_empty, time_col, freq, df_self.name, _skip_validate=True),
        DataFrame(df2_empty, time_col, freq, df_other.name, _skip_validate=True),
    )


def auto_align(
    method: Callable[..., "DataFrame"],
) -> Callable[..., Any]:
    """
    Decorator for operators (+, -, *, /).
    Includes a FAST PATH to skip alignment if DFs are already matched.
    Otherwise, enforces 'inner' alignment.
    """

    @wraps(method)
    def wrapper(self: "DataFrame", other: Any) -> Any:
        if not isinstance(other, DataFrame):
            return NotImplemented

        # If metadata matches perfectly, row N of self corresponds to row N of other.
        # We assume __init__ guaranteed dense data if freq is present.
        has_data = len(self._df) > 0 and len(other._df) > 0

        if (
            has_data
            and self.freq is not None
            and other.freq is not None
            and self.freq == other.freq
            and len(self._df) == len(other._df)
        ):
            # Check Start Time and End Time
            start_match = self._df[self.time_col][0] == other._df[other.time_col][0]
            end_match = self._df[self.time_col][-1] == other._df[other.time_col][-1]

            if start_match and end_match:
                # Skip align() overhead.
                return method(self, self, other)

        # Explicit Inner Align
        df1_aligned, df2_aligned = self.align(other, how="inner")

        # Handle empty intersection gracefully
        if len(df1_aligned._df) == 0:
            time_col = self.time_col
            other_time_col = other.time_col
            cols_self = set(self._df.columns)
            cols_other = set(other._df.columns)
            common_cols = sorted(
                list((cols_self & cols_other) - {time_col, other_time_col})
            )
            return _return_empty_aligned(
                self, other, method, common_cols, df1_aligned.freq
            )

        return method(self, df1_aligned, df2_aligned)

    return wrapper


class DataFrame:
    def __init__(
        self,
        df: pl.DataFrame,
        time_col: str,
        freq: int | None = None,
        name: str = "",
        _skip_validate: bool = False,
    ):
        self.time_col = time_col
        self.name = name

        if _skip_validate:
            self._df = df
            self.freq = freq
        else:
            # 1. Sort & Check (O(N) or O(N log N))
            # We assume user provides mostly clean data, but we must ensure Sort for Polars checks
            sorted_df = df.sort(time_col)

            # 2. Calculate Freq (O(N) scan)
            if freq is not None:
                self.freq = int(freq)
                # A forced freq must actually divide the observed spacing,
                # else off-grid points get silently relabeled onto grid slots.
                self._check_on_grid(sorted_df, self.freq)
            else:
                self.freq = self._calculate_freq(sorted_df)

            # 3. Upsample / Make Dense (O(N))
            # Critical: This guarantees that if two DFs have same start/end/freq, they map 1:1.
            if self.freq and len(sorted_df) > 1:
                self._df = self._ensure_dense(sorted_df, self.freq)
            else:
                self._df = sorted_df

    @property
    def value_col(self) -> str:
        """The single non-time value column.

        Series-shaped consumers (e.g. performance metrics on a pnl frame)
        use this to locate the value column by structure instead of a
        hardcoded name. Raises if the frame is not single-series.

        Raises
        ------
        ValueError
            If there is not exactly one non-time column.
        """
        cols = [c for c in self._df.columns if c != self.time_col]
        if len(cols) != 1:
            raise ValueError(f"expected a single value column, got {cols}")
        return cols[0]

    def copy(self) -> "DataFrame":
        return DataFrame(
            self._df.clone(),
            self.time_col,
            self.freq,
            self.name,
            _skip_validate=True,
        )

    def rename(self, mapping: dict[str, str], inplace: bool = False) -> "DataFrame":
        new_df = self._df.rename(mapping)
        new_time_col = mapping.get(self.time_col, self.time_col)
        new_name = f"{self.name}.rename({mapping})"

        if inplace:
            self._df = new_df
            self.time_col = new_time_col
            self.name = new_name
            return self
        else:
            return DataFrame(
                new_df, new_time_col, self.freq, new_name, _skip_validate=True
            )

    def shift_time(self, delta: timedelta) -> "DataFrame":
        """
        Shifts the underlying time column by a given timedelta.
        """
        new_df = self._df.with_columns(
            (pl.col(self.time_col) + delta).alias(self.time_col)
        )

        return DataFrame(
            new_df,
            self.time_col,
            self.freq,
            f"{self.name}.shift_time({delta!r})",
            _skip_validate=True,
        )

    def concat(self, other: "DataFrame", how: str = "inner") -> "DataFrame":
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"Can only concat with another DataFrame, got {type(other)}"
            )

        df_self_aligned, df_other_aligned = self.align(other, how=how)
        other_pl = df_other_aligned._df.drop(df_other_aligned.time_col)
        df: DataFrame = df_self_aligned.hstack(other_pl)
        df.name = f"{self.name}.concat({other.name}, how='{how}')"

        return df

    def align(
        self, other: "DataFrame", how: str = "inner"
    ) -> tuple["DataFrame", "DataFrame"]:
        """
        Explicitly align two DataFrames.
        :param how: 'inner' (Intersection) or 'outer' (Union + FFill)
        """
        if not isinstance(other, DataFrame):
            raise TypeError("Can only align with another DataFrame")

        time_col = self.time_col
        other_time_col = other.time_col

        if self.freq is None or other.freq is None:
            return self._empty(), other._empty()

        # No data on either side -> empty intersection (avoid indexing [0]/[-1])
        if len(self._df) == 0 or len(other._df) == 0:
            return self._empty(), other._empty()

        # Rename for consistency
        other_df_renamed = (
            other._df.rename({other_time_col: time_col})
            if other_time_col != time_col
            else other._df
        )

        start_self = self._df[time_col][0]
        end_self = self._df[time_col][-1]
        start_other = other_df_renamed[time_col][0]
        end_other = other_df_renamed[time_col][-1]

        # Determine Time Bounds
        if how == "outer":
            start_ts = min(start_self, start_other)
            end_ts = max(end_self, end_other)
        else:  # inner
            start_ts = max(start_self, start_other)
            end_ts = min(end_self, end_other)

        if start_ts > end_ts:
            return self._empty(), other._empty()

        # GCD Frequency
        if self.freq == other.freq:
            gcd_freq_s = self.freq
        else:
            gcd_freq_s = math.gcd(self.freq, other.freq)

        # Create Grid
        grid_lf = (
            pl.datetime_range(
                start=start_ts,
                end=end_ts,
                interval=timedelta(seconds=gcd_freq_s),
                eager=True,
            )
            .alias(time_col)
            .to_frame()
            .lazy()
        )

        # Join & Fill Logic
        def _process(base_lazy: pl.LazyFrame, original_freq: int) -> pl.DataFrame:
            # Bound the backfill to one source bar (at gcd resolution) for both
            # how modes. outer extends the grid past a frame's range, so an
            # unbounded fill would fabricate that frame's tail from its last value.
            tol = timedelta(seconds=original_freq - gcd_freq_s)

            return grid_lf.join_asof(
                base_lazy, on=time_col, strategy="backward", tolerance=tol
            ).collect()

        df1_aligned = _process(self._df.lazy(), self.freq)
        df2_aligned = _process(other_df_renamed.lazy(), other.freq)

        return (
            DataFrame(
                df1_aligned, time_col, gcd_freq_s, self.name, _skip_validate=True
            ),
            DataFrame(
                df2_aligned, time_col, gcd_freq_s, other.name, _skip_validate=True
            ),
        )

    def _calculate_freq(self, df: pl.DataFrame) -> int | None:
        if len(df) < 2:
            return None
        diffs = df.select(pl.col(self.time_col).diff()).to_series().drop_nulls()
        min_diff = diffs.min()
        assert isinstance(min_diff, timedelta)
        if min_diff.total_seconds() == 0:
            raise ValueError("Duplicate timestamps detected in DataFrame.")
        if int(min_diff.total_seconds()) == 0:
            raise ValueError("Minimum time difference is less than 1 second.")

        # freq = gcd of all gaps, so every timestamp lands on the grid even when
        # the smallest observed gap isn't itself the base period (120s & 180s -> 60s).
        diffs_sec = diffs.dt.total_seconds().cast(pl.Int64).to_list()
        return int(reduce(math.gcd, diffs_sec))

    def _check_on_grid(self, df: pl.DataFrame, freq: int) -> None:
        if len(df) < 2:
            return
        offsets = df.select(
            (pl.col(self.time_col) - pl.col(self.time_col).first())
            .dt.total_seconds()
            .cast(pl.Int64)
            .alias("off")
        )["off"]
        if (offsets % freq != 0).any():
            raise ValueError(
                f"Timestamps are not aligned to the provided freq={freq}s grid."
            )

    def _ensure_dense(self, df: pl.DataFrame, freq: int) -> pl.DataFrame:
        start = df[self.time_col][0]
        end = df[self.time_col][-1]
        grid = (
            pl.datetime_range(
                start=start, end=end, interval=timedelta(seconds=freq), eager=True
            )
            .alias(self.time_col)
            .to_frame()
        )
        # Bound the forward-fill to one bar so gaps larger than `freq` stay null
        # instead of being fabricated from the last known value.
        return grid.join_asof(
            df,
            on=self.time_col,
            strategy="backward",
            tolerance=timedelta(seconds=freq),
        )

    def _empty(self) -> "DataFrame":
        return DataFrame(
            self._df.clear(), self.time_col, self.freq, self.name, _skip_validate=True
        )

    def select(self, *exprs: Any, **named_exprs: Any) -> "DataFrame":
        return DataFrame(
            self._df.select(*exprs, **named_exprs),
            self.time_col,
            self.freq,
            self.name,
            _skip_validate=True,
        )

    def with_columns(self, *exprs: Any, **named_exprs: Any) -> "DataFrame":
        return DataFrame(
            self._df.with_columns(*exprs, **named_exprs),
            self.time_col,
            self.freq,
            self.name,
            _skip_validate=True,
        )

    @property
    def columns(self) -> list[str]:
        return self._df.columns

    def __str__(self) -> str:
        freq_str = f"{self.freq}s" if self.freq is not None else "None"
        return f"DataFrame(name={self.name}, rows={len(self._df)}, freq={freq_str})\n{self._df.__repr__()}"

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._df, name)
        if callable(attr):

            def proxy(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                if isinstance(result, pl.DataFrame):
                    return DataFrame(
                        result, self.time_col, self.freq, self.name, _skip_validate=True
                    )
                return result

            return proxy
        return attr

    @auto_align
    def _add_aligned(self, df1: "DataFrame", df2: "DataFrame") -> "DataFrame":
        cols1 = set(df1._df.columns) - {self.time_col}
        cols2 = set(df2._df.columns) - {self.time_col}
        common_cols = sorted(list(cols1 & cols2))

        res = df1._df.select(
            pl.col(self.time_col),
            *[(pl.col(c) + df2._df[c]).alias(c) for c in common_cols],
        )
        return DataFrame(
            res,
            self.time_col,
            df1.freq,
            f"({df1.name} + {df2.name})",
            _skip_validate=True,
        )

    @auto_align
    def _sub_aligned(self, df1: "DataFrame", df2: "DataFrame") -> "DataFrame":
        cols1 = set(df1._df.columns) - {self.time_col}
        cols2 = set(df2._df.columns) - {self.time_col}
        common_cols = sorted(list(cols1 & cols2))

        res = df1._df.select(
            pl.col(self.time_col),
            *[(pl.col(c) - df2._df[c]).alias(c) for c in common_cols],
        )
        return DataFrame(
            res,
            self.time_col,
            df1.freq,
            f"({df1.name} - {df2.name})",
            _skip_validate=True,
        )

    @auto_align
    def _mul_aligned(self, df1: "DataFrame", df2: "DataFrame") -> "DataFrame":
        cols1 = set(df1._df.columns) - {self.time_col}
        cols2 = set(df2._df.columns) - {self.time_col}
        common_cols = sorted(list(cols1 & cols2))

        res = df1._df.select(
            pl.col(self.time_col),
            *[(pl.col(c) * df2._df[c]).alias(c) for c in common_cols],
        )
        return DataFrame(
            res,
            self.time_col,
            df1.freq,
            f"({df1.name} * {df2.name})",
            _skip_validate=True,
        )

    @auto_align
    def _truediv_aligned(self, df1: "DataFrame", df2: "DataFrame") -> "DataFrame":
        cols1 = set(df1._df.columns) - {self.time_col}
        cols2 = set(df2._df.columns) - {self.time_col}
        common_cols = sorted(list(cols1 & cols2))

        res = df1._df.select(
            pl.col(self.time_col),
            *[(pl.col(c) / df2._df[c]).alias(c) for c in common_cols],
        )
        return DataFrame(
            res,
            self.time_col,
            df1.freq,
            f"({df1.name} / {df2.name})",
            _skip_validate=True,
        )

    def __add__(self, other: Any) -> "DataFrame":
        if isinstance(other, DataFrame):
            result: DataFrame = self._add_aligned(other)
            return result
        res = self._df.select(
            pl.col(self.time_col), (pl.exclude(self.time_col) + other).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({self.name} + {str(other)})",
            _skip_validate=True,
        )

    def __radd__(self, other: Any) -> "DataFrame":
        res = self._df.select(
            pl.col(self.time_col), (other + pl.exclude(self.time_col)).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({str(other)} + {self.name})",
            _skip_validate=True,
        )

    def __sub__(self, other: Any) -> "DataFrame":
        if isinstance(other, DataFrame):
            result: DataFrame = self._sub_aligned(other)
            return result
        res = self._df.select(
            pl.col(self.time_col), (pl.exclude(self.time_col) - other).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({self.name} - {str(other)})",
            _skip_validate=True,
        )

    def __rsub__(self, other: Any) -> "DataFrame":
        res = self._df.select(
            pl.col(self.time_col), (other - pl.exclude(self.time_col)).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({str(other)} - {self.name})",
            _skip_validate=True,
        )

    def __mul__(self, other: Any) -> "DataFrame":
        if isinstance(other, DataFrame):
            result: DataFrame = self._mul_aligned(other)
            return result
        res = self._df.select(
            pl.col(self.time_col), (pl.exclude(self.time_col) * other).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({self.name} * {str(other)})",
            _skip_validate=True,
        )

    def __rmul__(self, other: Any) -> "DataFrame":
        res = self._df.select(
            pl.col(self.time_col), (other * pl.exclude(self.time_col)).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({str(other)} * {self.name})",
            _skip_validate=True,
        )

    def __truediv__(self, other: Any) -> "DataFrame":
        if isinstance(other, DataFrame):
            result: DataFrame = self._truediv_aligned(other)
            return result
        res = self._df.select(
            pl.col(self.time_col), (pl.exclude(self.time_col) / other).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({self.name} / {str(other)})",
            _skip_validate=True,
        )

    def __rtruediv__(self, other: Any) -> "DataFrame":
        res = self._df.select(
            pl.col(self.time_col), (other / pl.exclude(self.time_col)).name.keep()
        )
        return DataFrame(
            res,
            self.time_col,
            self.freq,
            f"({str(other)} / {self.name})",
            _skip_validate=True,
        )

    def abs(self) -> "DataFrame":
        return DataFrame(
            self._df.select(
                pl.col(self.time_col), pl.exclude(self.time_col).abs().name.keep()
            ),
            self.time_col,
            self.freq,
            f"{self.name}.abs()",
            _skip_validate=True,
        )

    def __abs__(self) -> "DataFrame":
        return self.abs()

    def __neg__(self) -> "DataFrame":
        return DataFrame(
            self._df.select(pl.col(self.time_col), -pl.exclude(self.time_col)),
            self.time_col,
            self.freq,
            f"-{self.name}",
            _skip_validate=True,
        )
