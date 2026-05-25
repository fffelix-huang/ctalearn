from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from ctalearn.core.dataframe import DataFrame
from ctalearn.operator._time_series import (
    ts_corr,
    ts_decay_linear,
    ts_delay,
    ts_delta,
    ts_ffill,
    ts_hurst_exponent,
    ts_max,
    ts_mean,
    ts_median,
    ts_min,
    ts_rank,
    ts_robust_zscore,
    ts_scale,
    ts_std_dev,
    ts_sum,
    ts_zscore,
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


class TestTsRank:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0]})
        result = ts_rank(df, window=3)
        expected = _create_df(
            {"val": [None, None, 1.0, 0.5, 1.0, 0.0]},
            name="ts_rank(df, 3, constant=0.0)",
        )
        assert_df_equal(result, expected)

    def test_with_constant(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0]})
        result = ts_rank(df, window=3, constant=0.5)
        expected = _create_df(
            {"val": [None, None, 1.5, 1.0, 1.5, 0.5]},
            name="ts_rank(df, 3, constant=0.5)",
        )
        assert_df_equal(result, expected)

    def test_same_value(self) -> None:
        df = _create_df({"val": [1.0, 1.0, 1.0, 2.0, 2.0]})
        result = ts_rank(df, window=3)
        expected = _create_df(
            {"val": [None, None, 0.5, 1.0, 0.75]}, name="ts_rank(df, 3, constant=0.0)"
        )
        assert_df_equal(result, expected)

    def test_window_equals_1(self) -> None:
        df = _create_df({"val": [1.0, 1.0, 1.0, 2.0, 2.0]})
        result = ts_rank(df, window=1)
        expected = _create_df(
            {"val": [0.5, 0.5, 0.5, 0.5, 0.5]}, name="ts_rank(df, 1, constant=0.0)"
        )
        assert_df_equal(result, expected)

    def test_window_equals_1_with_constant(self) -> None:
        # window==1 must honor `constant` just like window>=2 does.
        df = _create_df({"val": [1.0, 2.0, 3.0, 4.0]})
        result = ts_rank(df, window=1, constant=0.5)
        expected = _create_df(
            {"val": [1.0, 1.0, 1.0, 1.0]}, name="ts_rank(df, 1, constant=0.5)"
        )
        assert_df_equal(result, expected)

    def test_window_equals_1_nan(self) -> None:
        # NaN input must become null, not a leaked float NaN.
        df = _create_df({"val": [1.0, float("nan"), 3.0]})
        result = ts_rank(df, window=1)
        expected = _create_df(
            {"val": [0.5, None, 0.5]}, name="ts_rank(df, 1, constant=0.0)"
        )
        assert_df_equal(result, expected)

    def test_multi_dims(self) -> None:
        data = {
            "c1": [1.0, 2.0, 3.0, 5.0, 3.0, None],
            "c2": [2.0, 1.0, 3.0, 6.0, 1.0, 4.0],
            "c3": [3.0, 3.0, 1.0, 7.0, 1.0, 5.0],
        }
        df = _create_df(data)
        result = ts_rank(df, window=4)

        expected_data = {
            "c1": [None, None, None, 1.0, 0.5, None],
            "c2": [None, None, None, 1.0, 1 / 6, 2 / 3],
            "c3": [None, None, None, 1.0, 1 / 6, 2 / 3],
        }
        expected = _create_df(expected_data, name="ts_rank(df, 4, constant=0.0)")
        assert_df_equal(result, expected)

    def test_nan(self) -> None:
        df = _create_df({"val": [1.0, None, 4.0, 3.0, 5.0, 2.0]})
        result = ts_rank(df, window=3)
        # In Polars rolling map, if input window contains null, our _rank_func returns null
        expected = _create_df(
            {"val": [None, None, None, None, 1.0, 0.0]},
            name="ts_rank(df, 3, constant=0.0)",
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_rank("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_rank(df, window=0)

    @pytest.mark.benchmark(group="ts_rank", min_rounds=30, disable_gc=True)
    def test_benchmark_50000(self, benchmark: Any) -> None:
        np.random.seed(42)
        df = _create_df({"c1": np.random.randn(50000), "c2": np.random.randn(50000)})
        _ = benchmark(ts_rank, df, window=500)


class TestTsMean:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0]})
        result = ts_mean(df, window=3)
        expected = _create_df(
            {"val": [None, None, 7 / 3, 9 / 3, 12 / 3, 10 / 3]}, name="ts_mean(df, 3)"
        )
        assert_df_equal(result, expected)

    def test_multi_dims(self) -> None:
        df = _create_df(
            {
                "c1": [1.0, 2.0, 3.0, 5.0, 3.0, None],
                "c2": [2.0, 1.0, 3.0, 6.0, 1.0, 4.0],
                "c3": [3.0, 3.0, 1.0, 7.0, 1.0, 5.0],
            }
        )
        result = ts_mean(df, window=4)
        expected = _create_df(
            {
                "c1": [None, None, None, 11 / 4, 13 / 4, None],
                "c2": [None, None, None, 12 / 4, 11 / 4, 14 / 4],
                "c3": [None, None, None, 14 / 4, 12 / 4, 14 / 4],
            },
            name="ts_mean(df, 4)",
        )
        assert_df_equal(result, expected)

    def test_window_equals_1(self) -> None:
        df = _create_df({"val": [1.0, 1.0, 1.0, 2.0, 2.0]})
        result = ts_mean(df, window=1)
        # ts_mean with window=1 is essentially the value itself (float)
        # Note: Polars rolling_mean returns float, inputs are int, so we expect casts
        expected = _create_df({"val": [1.0, 1.0, 1.0, 2.0, 2.0]}, name="ts_mean(df, 1)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_mean("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_mean(df, window=0)


class TestTsMedian:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0]})
        result = ts_median(df, window=3)
        expected = _create_df(
            {"val": [None, None, 2.0, 3.0, 4.0, 3.0]}, name="ts_median(df, 3)"
        )
        assert_df_equal(result, expected)

    def test_multi_dims(self) -> None:
        df = _create_df(
            {
                "c1": [1.0, 2.0, 3.0, 5.0, 3.0, None],
                "c2": [2.0, 1.0, 3.0, 6.0, 1.0, 4.0],
                "c3": [3.0, 3.0, 1.0, 7.0, 1.0, 5.0],
            }
        )
        result = ts_median(df, window=4)
        expected = _create_df(
            {
                "c1": [None, None, None, 2.5, 3.0, None],
                "c2": [None, None, None, 2.5, 2.0, 3.5],
                "c3": [None, None, None, 3.0, 2.0, 3.0],
            },
            name="ts_median(df, 4)",
        )
        assert_df_equal(result, expected)

    def test_window_equals_1(self) -> None:
        df = _create_df({"val": [1.0, 1.0, 1.0, 2.0, 2.0]})
        result = ts_median(df, window=1)
        expected = _create_df(
            {"val": [1.0, 1.0, 1.0, 2.0, 2.0]}, name="ts_median(df, 1)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_median("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_median(df, window=0)


class TestTsStdDev:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0]})
        result = ts_std_dev(df, window=3)
        expected = _create_df(
            {"val": [None, None, 1.24721913, 0.81649658, 0.81649658, 1.24721913]},
            name="ts_std_dev(df, 3, ddof=0)",
        )
        assert_df_equal(result, expected)

    def test_window_equals_1(self) -> None:
        df = _create_df({"val": [1.0, 1.0, 1.0, 2.0, 2.0]})
        result = ts_std_dev(df, window=1)
        expected = _create_df(
            {"val": [0.0, 0.0, 0.0, 0.0, 0.0]}, name="ts_std_dev(df, 1, ddof=0)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_std_dev("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_std_dev(df, window=0)


class TestTsZscore:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0]})
        result = ts_zscore(df, window=3)
        expected = _create_df(
            {"val": [None, None, 1.33630621, 0.0, 1.22474487, -1.06904497]},
            name="ts_zscore(df, 3)",
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_zscore("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_zscore(df, window=0)


class TestTsRobustZscore:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 3.0, 10.0, 10.0, None]})
        result = ts_robust_zscore(df, window=3)

        val_0 = (3 - 2) / (1.4826 * 1)
        val_1 = (10 - 3) / (1.4826 * 1)

        expected = _create_df(
            {"val": [None, None, val_0, val_1, 0.0, None]},
            name="ts_robust_zscore(df, 3)",
        )

        assert_df_equal(result, expected)

    def test_window_equals_1(self) -> None:
        # Window=1 implies Median=Val, AbsDev=0, MAD=0 -> Result 0
        df = _create_df({"val": [10.0, 20.0, 30.0]})
        result = ts_robust_zscore(df, window=1)
        expected = _create_df({"val": [0.0, 0.0, 0.0]}, name="ts_robust_zscore(df, 1)")
        assert_df_equal(result, expected)

    def test_multi_dims(self) -> None:
        data = {"c1": [5.0, 5.0, 5.0, 5.0], "c2": [1.0, 2.0, 3.0, 4.0]}
        df = _create_df(data)
        result = ts_robust_zscore(df, window=3)

        C = 1.4826
        expected_data = {"c1": [None, None, 0.0, 0.0], "c2": [None, None, 1 / C, 1 / C]}
        expected = _create_df(expected_data, name="ts_robust_zscore(df, 3)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_robust_zscore("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_robust_zscore(df, window=0)

    @pytest.mark.benchmark(group="ts_robust_zscore", min_rounds=30, disable_gc=True)
    def test_benchmark_50000(self, benchmark: Any) -> None:
        np.random.seed(42)
        df = _create_df({"c1": np.random.randn(50000), "c2": np.random.randn(50000)})
        _ = benchmark(ts_robust_zscore, df, window=500)


class TestTsSum:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0, None]})
        result = ts_sum(df, window=3)
        expected = _create_df(
            {"val": [None, None, 7.0, 9.0, 12.0, 10.0, None]}, name="ts_sum(df, 3)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_sum("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_sum(df, window=0)


class TestTsMin:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0, None]})
        result = ts_min(df, window=3)
        expected = _create_df(
            {"val": [None, None, 1.0, 2.0, 3.0, 2.0, None]}, name="ts_min(df, 3)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_min("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_min(df, window=0)


class TestTsMax:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0, None]})
        result = ts_max(df, window=3)
        expected = _create_df(
            {"val": [None, None, 4.0, 4.0, 5.0, 5.0, None]}, name="ts_max(df, 3)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_max("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_max(df, window=0)


class TestTsScale:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0, None]})
        result = ts_scale(df, window=3)
        expected = _create_df(
            {"val": [None, None, 1.0, 0.5, 1.0, 0.0, None]},
            name="ts_scale(df, 3, constant=0.0)",
        )
        assert_df_equal(result, expected)

    def test_all_same(self) -> None:
        df = _create_df({"val": [1.0, 1.0, 1.0]})
        result = ts_scale(df, window=2)
        expected = _create_df(
            {"val": [None, 0.0, 0.0]}, name="ts_scale(df, 2, constant=0.0)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_scale("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_scale(df, window=0)


class TestTsDecayLinear:
    def test_basic(self) -> None:
        df = _create_df({"val": [6.0, 5.0, 4.0, 5.0, 30.0, 3.0]})
        result = ts_decay_linear(df, window=5)
        expected = _create_df(
            {"val": [None, None, None, None, 13.2, 10.8666666]},
            name="ts_decay_linear(df, 5, dense=True)",
        )
        assert_df_equal(result, expected)

    def test_dense_true(self) -> None:
        df = _create_df({"val": [1.0, None, 4.0, 3.0, 5.0]})
        result = ts_decay_linear(df, window=3, dense=True)
        expected = _create_df(
            {"val": [None, None, None, None, (4.0 * 1 + 3.0 * 2 + 5.0 * 3) / 6]},
            name="ts_decay_linear(df, 3, dense=True)",
        )
        assert_df_equal(result, expected)

    def test_dense_false(self) -> None:
        df = _create_df({"val": [1.0, None, 4.0, 3.0]})
        result = ts_decay_linear(df, window=3, dense=False)
        expected = _create_df(
            {
                "val": [
                    None,
                    None,
                    (1.0 + 0.0 * 2 + 4.0 * 3) / 6,
                    (0.0 + 4.0 * 2 + 3.0 * 3) / 6,
                ]
            },
            name="ts_decay_linear(df, 3, dense=False)",
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_decay_linear("invalid_type", window=3)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_decay_linear(df, window=0)


class TestTsDelay:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, -4.0, -3.0, 0.0, 2.0, None]})
        result = ts_delay(df, d=2)
        expected = _create_df(
            {"val": [None, None, 1.0, 2.0, -4.0, -3.0, 0.0]}, name="ts_delay(df, 2)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_delay("invalid_type", d=2)  # type: ignore[arg-type]

    def test_invalid_d(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`d` must be a positive integer"):
            ts_delay(df, d=0)


class TestTsDelta:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, -4.0, -3.0, 0.0, 2.0, None]})
        result = ts_delta(df, d=2)
        expected = _create_df(
            {"val": [None, None, -5.0, -5.0, 4.0, 5.0, None]}, name="ts_delta(df, 2)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_delta("invalid_type", d=2)  # type: ignore[arg-type]

    def test_invalid_d(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`d` must be a positive integer"):
            ts_delta(df, d=0)


class TestTsFfill:
    def test_basic_with_limit(self) -> None:
        df = _create_df({"val": [6.0, None, None, None, 30.0]})
        result = ts_ffill(df, limit=2)
        expected = _create_df(
            {"val": [6.0, 6.0, 6.0, None, 30.0]}, name="ts_ffill(df, limit=2)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_ffill("invalid_type")  # type: ignore[arg-type]

    def test_invalid_limit(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`limit` must be a positive integer"):
            ts_ffill(df, limit=0)


class TestTsCorr:
    def test_positive_corr(self) -> None:
        df_a = _create_df({"val": [1, 2, 3, 4]}, name="df_a")
        df_b = _create_df({"val": [5, 10, 15, 20]}, name="df_b")
        result = ts_corr(df_a, df_b, window=3)
        expected = _create_df(
            {"val": [None, None, 1.0, 1.0]}, name="ts_corr(df_a, df_b, 3)"
        )
        assert_df_equal(result, expected)

    def test_negative_corr(self) -> None:
        df_a = _create_df({"val": [1, 2, 3, 4]}, name="df_a")
        df_b = _create_df({"val": [-100, -200, -300, -400]}, name="df_b")
        result = ts_corr(df_a, df_b, window=3)
        expected = _create_df(
            {"val": [None, None, -1.0, -1.0]}, name="ts_corr(df_a, df_b, 3)"
        )
        assert_df_equal(result, expected)

    def test_multi_col(self) -> None:
        df_a = _create_df({"a": [1, 3, 5, 7, 9], "b": [1, 2, 3, 4, 5]}, name="df_a")
        df_b = _create_df(
            {"a": [1, 2, 4, 5, 6], "b": [10, None, 12, 13, 14]}, name="df_b"
        )
        result = ts_corr(df_a, df_b, window=3)
        expected = _create_df(
            {
                "a": [None, None, 0.981981, 0.981981, 1.0],
                "b": [None, None, None, None, 1.0],
            },
            name="ts_corr(df_a, df_b, 3)",
        )
        assert_df_equal(result, expected)

    def test_not_aligned(self) -> None:
        df_a = _create_df(
            {"a": [1, 3, 5, 7, 9, 1000], "b": [1, 2, 3, 4, 5, -1000]}, name="df_a"
        )
        df_b = _create_df(
            {"a": [1, 2, 4, 5, 6], "b": [10, None, 12, 13, 14]}, name="df_b"
        )
        result = ts_corr(df_a, df_b, window=3)
        expected = _create_df(
            {
                "a": [None, None, 0.981981, 0.981981, 1.0],
                "b": [None, None, None, None, 1.0],
            },
            name="ts_corr(df_a, df_b, 3)",
        )
        assert_df_equal(result, expected)

    def test_invalid_df_a(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(TypeError, match="`df_a` should be DataFrame"):
            ts_corr("invalid_type", df, window=2)  # type: ignore[arg-type]

    def test_invalid_df_b(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(TypeError, match="`df_b` should be DataFrame"):
            ts_corr(df, "invalid_type", window=2)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_corr(df, df, window=0)

    def test_no_common_columns(self) -> None:
        # Disjoint column names -> empty result with only the time column.
        df_a = _create_df({"a": [1.0, 2.0, 3.0]}, name="df_a")
        df_b = _create_df({"b": [1.0, 2.0, 3.0]}, name="df_b")
        result = ts_corr(df_a, df_b, window=2)
        assert result._df.columns == ["ts"]
        assert len(result._df) == 0

    def test_different_time_col(self) -> None:
        # df_b carries a different time column -> exercises the left_on/right_on join.
        ts = pl.datetime_range(
            datetime(2023, 1, 1),
            datetime(2023, 1, 1) + timedelta(seconds=3),
            interval="1s",
            eager=True,
        )
        df_a = _create_df({"val": [1.0, 2.0, 3.0, 4.0]}, name="df_a")
        df_b = DataFrame(
            pl.DataFrame({"val": [5.0, 10.0, 15.0, 20.0], "ts2": ts}),
            "ts2",
            name="df_b",
        )
        result = ts_corr(df_a, df_b, window=3)
        assert result.time_col == "ts"
        assert np.isclose(result._df["val"][2], 1.0)


class TestTsHurstExponent:
    def numpy_hurst_exponent(self, arr: np.ndarray, lags: tuple[int, int]) -> float:
        lag_start, lag_end = lags
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size < 2 or not np.isfinite(arr).all():
            return float("nan")

        end_excl = min(lag_end, int(arr.size))
        if end_excl <= lag_start:
            return float("nan")

        tau_list: list[float] = []
        lagvec_list: list[int] = []
        for lag in range(lag_start, end_excl):
            pp = np.subtract(arr[lag:], arr[:-lag])
            lagvec_list.append(lag)
            tau_list.append(float(np.std(pp)))

        if len(tau_list) < 2:
            return float("nan")

        tau = np.asarray(tau_list, dtype=np.float64)
        lagvec = np.asarray(lagvec_list, dtype=np.float64)
        if not np.isfinite(tau).all() or (tau <= 0).any():
            return float("nan")

        with np.errstate(divide="ignore", invalid="ignore"):
            m0 = np.polyfit(np.log10(lagvec), np.log10(tau), 1)[0]
        return float(m0) if np.isfinite(m0) else float("nan")

    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, 2.0, 4.0, 3.0, 5.0, 2.0]})
        # range(start, end) semantics (end exclusive), so (2, 4) tests lags 2 and 3.
        result = ts_hurst_exponent(df, window=5, lag_start=2, lag_end=4)

        expected = _create_df(
            {"val": [None, None, None, None, -1.564266937027183, 2.4050953588889366]},
            name="ts_hurst_exponent(df, 5, 2, 4)",
        )

        assert_df_equal(result, expected)

    def test_large_random(self) -> None:
        for seed in range(20):
            np.random.seed(seed)
            window = 20
            lag_start, lag_end = 2, 15
            values = np.random.randn(200)

            df = _create_df({"val": values})
            result = ts_hurst_exponent(
                df, window=window, lag_start=lag_start, lag_end=lag_end
            )

            expected_vals: list[float | None] = [None] * len(values)
            for i in range(window - 1, len(values)):
                w = np.array(values[i - window + 1 : i + 1], dtype=np.float64)
                v = self.numpy_hurst_exponent(w, (lag_start, lag_end))
                expected_vals[i] = None if np.isnan(v) else v

            expected = _create_df(
                {"val": expected_vals},
                name=f"ts_hurst_exponent(df, {window}, {lag_start}, {lag_end})",
            )

            assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            ts_hurst_exponent("invalid_type", window=5)  # type: ignore[arg-type]

    def test_invalid_window(self) -> None:
        df = _create_df({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="`window` must be a positive integer"):
            ts_hurst_exponent(df, window=0)

    def test_invalid_lags_order(self) -> None:
        df = _create_df({"val": [1.0] * 10})
        with pytest.raises(ValueError, match="0 < start <= end"):
            ts_hurst_exponent(df, window=10, lag_start=5, lag_end=3)

    def test_lags_too_narrow(self) -> None:
        df = _create_df({"val": [1.0] * 10})
        with pytest.raises(ValueError, match="all values will be null"):
            ts_hurst_exponent(df, window=5, lag_start=2, lag_end=3)

    @pytest.mark.benchmark(group="ts_hurst_exponent", disable_gc=True)
    def test_benchmark_2000(self, benchmark: Any) -> None:
        # lag_end < window so every lag is usable (lag_end == window would hit
        # the all-null early return and measure nothing). 2000 bars x 2 cols.
        np.random.seed(42)
        df = _create_df({"c1": np.random.randn(2000), "c2": np.random.randn(2000)})
        _ = benchmark(ts_hurst_exponent, df, window=50, lag_start=2, lag_end=40)
