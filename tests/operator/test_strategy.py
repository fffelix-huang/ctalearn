from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from ctalearn.core.dataframe import DataFrame
from ctalearn.operator._strategy import (
    trend,
    trend_fast,
    trend_mean_reversion,
    trend_reverse,
    trend_reverse_fast,
    trend_reverse_mean_reversion,
    trend_time,
)
from ctalearn.testing import assert_df_equal


def _create_df(data: dict[str, Any], name: str = "df") -> DataFrame:
    length = len(data[next(iter(data))])
    dates = pl.datetime_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    return DataFrame(pl.DataFrame({**data, "ts": dates}), "ts", name=name)


class TestTrend:
    def test_basic(self) -> None:
        df = _create_df({"val": [0.0, 10.0, 5.0, -10.0, -5.0, 12.0]})
        result = trend(df, threshold=10.0)
        expected = _create_df(
            {"val": [None, 1.0, 1.0, -1.0, -1.0, 1.0]}, name="trend(df, 10.0)"
        )
        assert_df_equal(result, expected)

    def test_nan_handling(self) -> None:
        df = _create_df({"val": [10.0, None, 10.0]})
        result = trend(df, threshold=5)
        expected = _create_df({"val": [1.0, 0.0, 1.0]}, name="trend(df, 5)")
        assert_df_equal(result, expected)

    def test_float_nan_handling(self) -> None:
        # NaN (not null) must map to 0, not leak as a long (NaN >= thr is True in Polars)
        df = _create_df({"val": [float("nan"), 2.0, 0.0, float("nan"), 0.0]})
        result = trend(df, threshold=1.0)
        expected = _create_df({"val": [0.0, 1.0, 1.0, 0.0, 0.0]}, name="trend(df, 1.0)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            trend("invalid_type", threshold=1.0)  # type: ignore[arg-type]

    def test_invalid_threshold(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="threshold must be greater than 0"):
            trend(df, threshold=0)


class TestTrendReverse:
    def test_basic(self) -> None:
        df = _create_df({"val": [0.0, 10.0, 5.0, -10.0]})
        result = trend_reverse(df, threshold=10.0)
        expected = _create_df({"val": [None, -1.0, -1.0, 1.0]}, name="-trend(df, 10.0)")
        assert_df_equal(result, expected)


class TestTrendMeanReversion:
    def test_basic(self) -> None:
        df = _create_df({"val": [15.0, 5.0, -5.0, -15.0, 15.0, -2.0, 2.0, 15.0]})
        result = trend_mean_reversion(df, threshold=10.0)
        expected = _create_df(
            {"val": [1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0]},
            name="trend_mean_reversion(df, 10.0)",
        )
        assert_df_equal(result, expected)

    def test_exact_zero_cross(self) -> None:
        df = _create_df({"val": [-15.0, 0.0, 5.0]})
        result = trend_mean_reversion(df, threshold=10.0)
        expected = _create_df(
            {"val": [-1.0, -1.0, -1.0]}, name="trend_mean_reversion(df, 10.0)"
        )
        assert_df_equal(result, expected)

    def test_nan_handling(self) -> None:
        df = _create_df({"val": [15.0, None, 15.0]})
        result = trend_mean_reversion(df, threshold=10.0)
        expected = _create_df(
            {"val": [1.0, 0.0, 1.0]}, name="trend_mean_reversion(df, 10.0)"
        )
        assert_df_equal(result, expected)

    def test_float_nan_handling(self) -> None:
        df = _create_df({"val": [15.0, float("nan"), 15.0]})
        result = trend_mean_reversion(df, threshold=10.0)
        expected = _create_df(
            {"val": [1.0, 0.0, 1.0]}, name="trend_mean_reversion(df, 10.0)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            trend_mean_reversion("invalid_type", threshold=1.0)  # type: ignore[arg-type]

    def test_invalid_threshold(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="threshold must be greater than 0"):
            trend_mean_reversion(df, threshold=0)


class TestTrendReverseMeanReversion:
    def test_basic(self) -> None:
        df = _create_df({"val": [15.0, -5.0, -15.0]})
        result = trend_reverse_mean_reversion(df, threshold=10.0)
        expected = _create_df(
            {"val": [-1.0, 0.0, 1.0]}, name="-trend_mean_reversion(df, 10.0)"
        )
        assert_df_equal(result, expected)


class TestTrendFast:
    def test_basic(self) -> None:
        df = _create_df(
            {"val": [0.0, 12.0, 8.0, 9.0, -12.0, -8.0, -9.0, 15.0, -15.0, -10.0, 9.0]}
        )
        result = trend_fast(df, threshold=10.0)
        expected = _create_df(
            {"val": [None, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0]},
            name="trend_fast(df, 10.0)",
        )
        assert_df_equal(result, expected)

    def test_nan_handling(self) -> None:
        df = _create_df({"val": [12.0, None, 12.0]})
        result = trend_fast(df, threshold=10.0)
        expected = _create_df({"val": [1.0, 0.0, 1.0]}, name="trend_fast(df, 10.0)")
        assert_df_equal(result, expected)

    def test_float_nan_handling(self) -> None:
        df = _create_df({"val": [12.0, float("nan"), 12.0]})
        result = trend_fast(df, threshold=10.0)
        expected = _create_df({"val": [1.0, 0.0, 1.0]}, name="trend_fast(df, 10.0)")
        assert_df_equal(result, expected)

    def test_close_threshold_cross(self) -> None:
        df = _create_df({"val": [12.0, 11.0, 9.0, 8.0]})
        result = trend_fast(df, threshold=10.0)
        expected = _create_df(
            {"val": [1.0, 1.0, 0.0, 0.0]}, name="trend_fast(df, 10.0)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            trend_fast("invalid_type", threshold=1.0)  # type: ignore[arg-type]

    def test_invalid_threshold(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="threshold must be greater than 0"):
            trend_fast(df, threshold=0)


class TestTrendReverseFast:
    def test_basic(self) -> None:
        df = _create_df(
            {"val": [0.0, 12.0, 8.0, 9.0, -12.0, -8.0, -9.0, 15.0, -15.0, -10.0, 9.0]}
        )
        result = trend_reverse_fast(df, threshold=10.0)
        expected = _create_df(
            {"val": [None, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0]},
            name="-trend_fast(df, 10.0)",
        )
        assert_df_equal(result, expected)


class TestTrendTime:
    def test_basic_long_and_short(self) -> None:
        df_long = _create_df(
            {
                "val": [
                    0.0,
                    10.0,
                    0.0,
                    0.0,
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                    0.0,
                    0.0,
                    0.0,
                    10.0,
                    -10.0,
                    0.0,
                    0.0,
                ]
            }
        )
        result_long = trend_time(df_long, threshold=10.0, step=3)
        expected_long = _create_df(
            {
                "val": [
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            },
            name="trend_time(df, 10.0, step=3)",
        )
        assert_df_equal(result_long, expected_long)

        df_short = _create_df({"val": [0.0, -10.0, 0.0, 0.0, 0.0]})
        result_short = trend_time(df_short, threshold=10.0, step=3)
        expected_short = _create_df(
            {"val": [0.0, -1.0, -1.0, -1.0, 0.0]}, name="trend_time(df, 10.0, step=3)"
        )
        assert_df_equal(result_short, expected_short)

    def test_nan_handling_does_not_break_window(self) -> None:
        df = _create_df({"val": [10.0, None, 0.0, 0.0]})
        result = trend_time(df, threshold=5.0, step=3)
        expected = _create_df(
            {"val": [1.0, 1.0, 1.0, 0.0]}, name="trend_time(df, 5.0, step=3)"
        )
        assert_df_equal(result, expected)

    def test_float_nan_does_not_trigger(self) -> None:
        # NaN must not create a trigger (NaN >= thr is True in Polars)
        df = _create_df({"val": [0.0, float("nan"), 0.0, 0.0]})
        result = trend_time(df, threshold=5.0, step=3)
        expected = _create_df(
            {"val": [0.0, 0.0, 0.0, 0.0]}, name="trend_time(df, 5.0, step=3)"
        )
        assert_df_equal(result, expected)

    def test_multi_column_independent(self) -> None:
        df = _create_df(
            {
                "a": [10.0, 0.0, 0.0, 0.0],
                "b": [0.0, 0.0, 10.0, 0.0],
            }
        )
        result = trend_time(df, threshold=10.0, step=2)
        expected = _create_df(
            {
                "a": [1.0, 1.0, 0.0, 0.0],
                "b": [0.0, 0.0, 1.0, 1.0],
            },
            name="trend_time(df, 10.0, step=2)",
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            trend_time("invalid_type", threshold=1.0, step=2)  # type: ignore[arg-type]

    def test_invalid_threshold(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="threshold must be greater than 0"):
            trend_time(df, threshold=0, step=2)

    def test_invalid_step(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="step must be greater than 0"):
            trend_time(df, threshold=1.0, step=0)
