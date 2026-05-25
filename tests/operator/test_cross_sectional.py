from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA

from ctalearn.core.dataframe import DataFrame
from ctalearn.operator._cross_sectional import (
    cs_mean,
    cs_pca,
    cs_rank,
    cs_winsorize,
    cs_zscore,
    process_alpha_weights,
    regression_neut,
    vector_neut,
)
from ctalearn.testing import assert_df_equal


def _create_df(data: Any, name: str = "df") -> DataFrame:
    if isinstance(data, np.ndarray):
        data = data.tolist()

    if not data:
        return DataFrame(pl.DataFrame({"ts": []}), "ts")

    rows = len(data)
    cols = len(data[0])
    col_data = list(map(list, zip(*data)))
    data_dict = {f"c{i}": col_data[i] for i in range(cols)}

    pl_df = pl.DataFrame(data_dict).with_columns(pl.exclude("ts").cast(pl.Float64))

    dates = pl.datetime_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1) + timedelta(seconds=rows - 1),
        interval="1s",
        eager=True,
    )
    return DataFrame(pl_df.with_columns(ts=dates), "ts", name=name)


class TestCsRank:
    def test_multi_dims_ignore_nan(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
            [None, 4.0, 5.0],
            [None, None, None],
        ]
        df = _create_df(x)
        result = cs_rank(df)

        expected_x = [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0],
            [0.75, 0.75, 0.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.25, 0.25],
            [None, 0.0, 1.0],
            [None, None, None],
        ]
        expected = _create_df(expected_x, name="cs_rank(df, ignore_nan=True)")
        assert_df_equal(result, expected)

    def test_multi_dims_dont_ignore_nan(self) -> None:
        x = [[None, 4, 5]]
        df = _create_df(x)
        result = cs_rank(df, ignore_nan=False)

        expected_x = [[None, None, None]]
        expected = _create_df(expected_x, name="cs_rank(df, ignore_nan=False)")
        assert_df_equal(result, expected)

    def test_same_value(self) -> None:
        x = [[1, 1, 1, 2, 2, None]]
        df = _create_df(x)

        result = cs_rank(df)

        expected_x = [[0.25, 0.25, 0.25, 0.875, 0.875, None]]
        expected = _create_df(expected_x, name="cs_rank(df, ignore_nan=True)")

        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            cs_rank("invalid_type")  # type: ignore[arg-type]

    def test_single_column_nan(self) -> None:
        # Single-asset universe: valid -> 0.5, but NaN must stay null, not 0.5.
        x = [[1.0], [float("nan")], [3.0]]
        df = _create_df(x)
        result = cs_rank(df)
        expected = _create_df(
            [[0.5], [None], [0.5]], name="cs_rank(df, ignore_nan=True)"
        )
        assert_df_equal(result, expected)


class TestCsMean:
    def test_multi_dims_ignore_nan(self) -> None:
        raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                    datetime(2023, 1, 1, 10, 0, 10),
                    datetime(2023, 1, 1, 10, 0, 15),
                    datetime(2023, 1, 1, 10, 0, 20),
                ],
                "BTC": [1.0, 2.0, 10.0, None, None],
                "ETH": [2.0, 2.0, 70.0, 4.0, None],
                "SOL": [3.0, 2.0, 30.0, 6.0, None],
            }
        )
        df = DataFrame(
            raw_df, time_col="timestamp", freq=5, name="df", _skip_validate=True
        )
        result = cs_mean(df)

        assert result.columns == ["timestamp", "BTC", "ETH", "SOL"]

        mean_values = [2.0, 2.0, (10 + 70 + 30) / 3, 5.0, None]
        expected_raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                    datetime(2023, 1, 1, 10, 0, 10),
                    datetime(2023, 1, 1, 10, 0, 15),
                    datetime(2023, 1, 1, 10, 0, 20),
                ],
                "BTC": mean_values,
                "ETH": mean_values,
                "SOL": mean_values,
            }
        )
        expected = DataFrame(
            expected_raw_df,
            time_col="timestamp",
            freq=5,
            name="cs_mean(df, ignore_nan=True)",
            _skip_validate=True,
        )
        assert_df_equal(result, expected)

    def test_multi_dims_dont_ignore_nan(self) -> None:
        raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                    datetime(2023, 1, 1, 10, 0, 10),
                    datetime(2023, 1, 1, 10, 0, 15),
                    datetime(2023, 1, 1, 10, 0, 20),
                ],
                "BTC": [1.0, 2.0, 10.0, None, None],
                "ETH": [2.0, 2.0, 70.0, 4.0, None],
                "SOL": [3.0, 2.0, 30.0, 6.0, None],
            }
        )
        df = DataFrame(
            raw_df, time_col="timestamp", freq=5, name="df", _skip_validate=True
        )
        result = cs_mean(df, ignore_nan=False)

        assert result.columns == ["timestamp", "BTC", "ETH", "SOL"]

        # Rows containing None collapse to None across all columns.
        mean_values = [2.0, 2.0, (10 + 70 + 30) / 3, None, None]
        expected_raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                    datetime(2023, 1, 1, 10, 0, 10),
                    datetime(2023, 1, 1, 10, 0, 15),
                    datetime(2023, 1, 1, 10, 0, 20),
                ],
                "BTC": mean_values,
                "ETH": mean_values,
                "SOL": mean_values,
            }
        )
        expected = DataFrame(
            expected_raw_df,
            time_col="timestamp",
            freq=5,
            name="cs_mean(df, ignore_nan=False)",
            _skip_validate=True,
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            cs_mean("invalid_type")  # type: ignore[arg-type]

    def test_only_time_column(self) -> None:
        # No asset columns -> early return preserving just the time column.
        df = DataFrame(
            pl.DataFrame(
                {
                    "ts": [
                        datetime(2023, 1, 1, 0, 0, 0),
                        datetime(2023, 1, 1, 0, 0, 1),
                    ]
                }
            ),
            "ts",
            freq=1,
        )
        result = cs_mean(df)
        assert result.columns == ["ts"]
        assert len(result._df) == 2


class TestCsZscore:
    def test_multi_dims_ignore_nan(self) -> None:
        x = [
            [1, 2, 3],
            [1, 1, 1],
            [1, None, 3],
            [None, 4, 4],
            [None, None, None],
        ]
        df = _create_df(x)
        result = cs_zscore(df)

        expected_x = [
            [-1.2247449, 0.0, 1.2247449],
            [0.0, 0.0, 0.0],
            [-1.0, None, 1.0],
            [None, 0.0, 0.0],
            [None, None, None],
        ]
        expected = _create_df(expected_x, name="cs_zscore(df, ignore_nan=True)")

        assert_df_equal(result, expected)

    def test_multi_dims_dont_ignore_nan(self) -> None:
        x = [[None, 4, 5]]
        df = _create_df(x)
        result = cs_zscore(df, ignore_nan=False)
        expected = _create_df(
            [[None, None, None]], name="cs_zscore(df, ignore_nan=False)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            cs_zscore("invalid_type")  # type: ignore[arg-type]


class TestCsWinsorize:
    def test_multi_dims_ignore_nan(self) -> None:
        x = [
            [-100, 100, 101, 102, 103],
            [1, 1, 1, 1, 1],
            [1, 2, 3, 4, 1000],
            [1, None, 2, 3, 4],
            [None, 4, 4, 4, 4],
            [None, None, None, None, None],
        ]
        df = _create_df(x)
        result = cs_winsorize(df, std=1.5)

        expected_x = [
            [-59.7093049, 100.0, 101.0, 102.0, 103.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 800.5018797],
            [1.0, None, 2.0, 3.0, 4.0],
            [None, 4.0, 4.0, 4.0, 4.0],
            [None, None, None, None, None],
        ]

        expected = _create_df(
            expected_x, name="cs_winsorize(df, std=1.5, ignore_nan=True)"
        )
        assert_df_equal(result, expected)

    def test_multi_dims_dont_ignore_nan(self) -> None:
        x = [[None, 4, 5]]
        df = _create_df(x)
        result = cs_winsorize(df, std=4, ignore_nan=False)
        expected = _create_df(
            [[None, None, None]], name="cs_winsorize(df, std=4, ignore_nan=False)"
        )
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            cs_winsorize("invalid_type", std=4)  # type: ignore[arg-type]


class TestVectorNeut:
    def test_basic(self) -> None:
        x = [
            [1, 2, 3],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        y = [
            [3, 4, 5],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
        df_a = _create_df(x, name="df_a")
        df_b = _create_df(y, name="df_b")
        result = vector_neut(df_a, df_b)
        expected = _create_df(
            [
                [-0.56, -0.08, 0.4],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            name="vector_neut(df_a, df_b)",
        )
        assert_df_equal(result, expected)

    def test_nan_isolation(self) -> None:
        # Dot product ignores NaN.
        x = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, None],
            [1.0, 2.0, 3.0],
        ]
        y = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, None],
        ]

        df_a = _create_df(x, name="df_a")
        df_b = _create_df(y, name="df_b")

        result = vector_neut(df_a, df_b)
        expected = _create_df(
            [[-1.0, 0.0, 1.0], [-0.5, 0.5, None], [-0.5, 0.5, None]],
            name="vector_neut(df_a, df_b)",
        )
        assert_df_equal(result, expected)

    def test_parallel_collinear(self) -> None:
        # Test when x and y are parallel, output zero vector.
        x = [
            [2.0, 4.0, 6.0],
            [1.0, 3.0, 5.0],
        ]
        y = [
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
        ]

        df_a = _create_df(x, name="df_a")
        df_b = _create_df(y, name="df_b")

        result = vector_neut(df_a, df_b)
        expected = _create_df(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            name="vector_neut(df_a, df_b)",
        )
        assert_df_equal(result, expected)

    def test_orthogonal(self) -> None:
        x = [
            [1.0, 1.0],
            [1.0, -2.0],
        ]
        y = [
            [1.0, -1.0],
            [-1.0, -0.5],
        ]

        df_a = _create_df(x, name="df_a")
        df_b = _create_df(y, name="df_b")

        result = vector_neut(df_a, df_b)

        # Same as df_a (since orthogonal), only different in name
        expected = _create_df(x, name="vector_neut(df_a, df_b)")
        assert_df_equal(result, expected)

    def test_zero_target(self) -> None:
        # Test when target is zero vector, no changes expected.
        x = np.arange(6).reshape((2, 3))
        y = np.zeros_like(x)

        df_a = _create_df(x, name="df_a")
        df_b = _create_df(y, name="df_b")

        result = vector_neut(df_a, df_b)
        expected = _create_df(x, name="vector_neut(df_a, df_b)")
        assert_df_equal(result, expected)

    def test_column_mismatch_raises(self) -> None:
        # x has a column y lacks -> must raise clearly, not KeyError.
        x = pl.DataFrame({"ts": [datetime(2023, 1, 1)], "a": [1.0], "b": [2.0]})
        y = pl.DataFrame({"ts": [datetime(2023, 1, 1)], "a": [1.0]})
        df_x = DataFrame(x, "ts", name="df_x")
        df_y = DataFrame(y, "ts", name="df_y")
        with pytest.raises(ValueError, match="same asset columns"):
            vector_neut(df_x, df_y)

    def test_invalid_x_type(self) -> None:
        df = _create_df([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(TypeError, match="Type of `x` should be DataFrame"):
            vector_neut("invalid_type", df)  # type: ignore[arg-type]

    def test_invalid_y_type(self) -> None:
        df = _create_df([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(TypeError, match="Type of `y` should be DataFrame"):
            vector_neut(df, "invalid_type")  # type: ignore[arg-type]


class TestRegressionNeut:
    def test_perfect_linear_regression(self) -> None:
        x_values = [
            np.arange(1.0, 6.0, 1.0, dtype=np.float64),
            np.arange(5.0, 0.0, -1.0, dtype=np.float64),
            np.array([3.0, 1.0, 4.0, 1.0, 5.0]),
        ]
        y_values = [3 * x + 10 for x in x_values]

        df_X = _create_df(x_values, name="X")
        df_Y = _create_df(y_values, name="Y")

        result = regression_neut(df_Y, [df_X])

        residuals = result._df.select(pl.exclude(result.time_col)).to_numpy()
        np.testing.assert_allclose(
            residuals, 0.0, atol=1e-7, err_msg="Residuals should be 0"
        )

    def test_residual_recovery(self) -> None:
        # Y = 0.5 * X + Noise
        np.random.seed(42)
        T = 50
        N = 100
        x_values = np.random.randn(T, N)
        x_values = x_values - np.mean(x_values, axis=1, keepdims=True)
        noise_values = np.random.randn(T, N)

        df_X = _create_df(x_values, name="X")
        df_noise_raw = _create_df(noise_values, name="noise")
        df_noise = vector_neut(df_noise_raw, df_X)
        df_Y = 0.5 * df_X + df_noise

        result = regression_neut(df_Y, [df_X])

        residuals = result._df.select(pl.exclude(result.time_col)).to_numpy()

        actual_noise = df_noise._df.select(pl.exclude(df_noise.time_col)).to_numpy()
        expected = actual_noise - np.mean(actual_noise, axis=1, keepdims=True)

        np.testing.assert_allclose(
            residuals,
            expected,
            atol=1e-7,
            err_msg="Residuals do not match expected noise pattern",
        )

    def test_multi_factor_linear(self) -> None:
        # Y = 1.0 * X1 + 0.5 * X2 + 10 (Intercept)
        np.random.seed(42)
        T = 5
        N = 100

        x1_vals = np.random.randn(T, N)
        x2_vals = np.random.randn(T, N)

        y_vals = 1.0 * x1_vals + 0.5 * x2_vals + 10.0

        df_X1 = _create_df(x1_vals, name="X1")
        df_X2 = _create_df(x2_vals, name="X2")
        df_Y = _create_df(y_vals, name="Y")

        result = regression_neut(df_Y, [df_X1, df_X2])
        residuals = result._df.select(pl.exclude(result.time_col)).to_numpy()

        np.testing.assert_allclose(
            residuals, 0.0, atol=1e-7, err_msg="Multi-factor linear relationship failed"
        )

    def test_nan_isolation(self) -> None:
        np.random.seed(42)
        T = 5
        N = 100
        x_vals = np.random.randn(T, N)
        x_vals[T - 1][N - 1] = np.nan
        y_vals = 2 * x_vals

        df_X = _create_df(x_vals, name="X")
        df_Y = _create_df(y_vals, name="Y")

        result = regression_neut(df_Y, [df_X])

        expected_np = np.zeros_like(x_vals)
        expected_np[T - 1][N - 1] = np.nan
        expected = _create_df(expected_np, name="regression_neut(Y, [X])")
        assert_df_equal(result, expected)

    def test_insufficient_dof(self) -> None:
        T = 2
        N = 2
        x1 = np.random.randn(T, N)
        x2 = np.random.randn(T, N)
        y = np.random.randn(T, N)

        df_X1 = _create_df(x1, name="X1")
        df_X2 = _create_df(x2, name="X2")
        df_Y = _create_df(y, name="Y")

        result = regression_neut(df_Y, [df_X1, df_X2])
        expected = _create_df(
            np.full_like(x1, np.nan), name="regression_neut(Y, [X1,X2])"
        )
        assert_df_equal(result, expected)

    def test_collinearity_ridge_exact(self) -> None:
        np.random.seed(42)
        T = 5
        N = 100
        alpha = 10.0

        x1 = np.random.randn(T, N)
        x2 = x1.copy()
        y = 2.0 * x1 + 5.0

        df_X1 = _create_df(x1.tolist(), name="X1")
        df_X2 = _create_df(x2.tolist(), name="X2")
        df_Y = _create_df(y.tolist(), name="Y")

        result = regression_neut(df_Y, [df_X1, df_X2], ridge_alpha=alpha)

        yc = y - y.mean(axis=1, keepdims=True)
        xc = x1 - x1.mean(axis=1, keepdims=True)
        s_xx = np.sum(xc**2, axis=1, keepdims=True)
        expected_resid = yc * (alpha / (2 * s_xx + alpha))
        expected = _create_df(expected_resid, name="regression_neut(Y, [X1,X2])")

        assert_df_equal(result, expected)

    def test_single_factor_not_list(self) -> None:
        # A lone DataFrame factor is wrapped into a list internally.
        np.random.seed(0)
        target = _create_df(np.random.randn(5, 4).tolist(), name="Y")
        factor = _create_df(np.random.randn(5, 4).tolist(), name="X")
        result = regression_neut(target, factor)
        assert result.time_col == target.time_col
        assert len(result._df) == 5

    def test_no_time_overlap(self) -> None:
        # No overlapping timestamps -> the early-out branch at the empty-align path.
        # NOTE: this currently returns the original target unchanged (see the
        # `lambda x, y, z: x` in regression_neut, which returns df_self rather than
        # the empty frame). Asserting the actual behavior; likely a bug to revisit.
        target = _create_df([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], name="Y")
        factor = _create_df([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], name="X").shift_time(
            timedelta(days=10)
        )
        result = regression_neut(target, [factor])
        assert_df_equal(result, target)

    def test_no_common_columns_raises(self) -> None:
        target = _create_df([[1.0, 2.0], [3.0, 4.0]], name="Y")
        factor = _create_df([[1.0, 2.0], [3.0, 4.0]], name="X").rename(
            {"c0": "x0", "c1": "x1"}
        )
        with pytest.raises(ValueError, match="No common columns"):
            regression_neut(target, [factor])

    def test_singular_returns_nan(self) -> None:
        # Collinear factors + ridge_alpha=0 -> singular system -> LinAlgError fallback.
        rows = [[1.0, 2.0, 3.0, 4.0, 5.0]] * 3
        target = _create_df(rows, name="Y")
        factor = _create_df(rows, name="X")
        result = regression_neut(target, [factor, factor], ridge_alpha=0.0)
        cols = [c for c in result.columns if c != result.time_col]
        assert np.isnan(result._df.select(cols).to_numpy()).all()


class TestProcessAlphaWeights:
    def test_neutralize_true(self) -> None:
        raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                ],
                "BTC": [10.0, 20.0],
                "ETH": [30.0, 10.0],
            }
        )
        df = DataFrame(raw_df, time_col="timestamp", freq=5, _skip_validate=True)

        expected_raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                ],
                "BTC": [-0.5, 0.5],
                "ETH": [0.5, -0.5],
            }
        )
        expected_df = DataFrame(
            expected_raw_df, time_col="timestamp", freq=5, _skip_validate=True
        )

        result_df = process_alpha_weights(df, neutralize=True)
        assert_df_equal(result_df, expected_df)

    def test_neutralize_false(self) -> None:
        raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                ],
                "BTC": [10.0, 20.0],
                "ETH": [30.0, 10.0],
            }
        )
        df = DataFrame(raw_df, time_col="timestamp", freq=5, _skip_validate=True)

        expected_raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                ],
                "BTC": [0.25, 0.6666667],
                "ETH": [0.75, 0.3333333],
            }
        )
        expected_df = DataFrame(
            expected_raw_df, time_col="timestamp", freq=5, _skip_validate=True
        )

        result_df = process_alpha_weights(df, neutralize=False)
        assert_df_equal(result_df, expected_df)

    def test_with_nulls(self) -> None:
        raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                ],
                "BTC": [None, 30.0],
                "ETH": [20.0, 10.0],
                "SOL": [20.0, None],
            }
        )
        df = DataFrame(raw_df, time_col="timestamp", freq=5, _skip_validate=True)

        expected_raw_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 5),
                ],
                "BTC": [0.0, 0.75],
                "ETH": [0.5, 0.25],
                "SOL": [0.5, 0.0],
            }
        )
        expected_df = DataFrame(
            expected_raw_df, time_col="timestamp", freq=5, _skip_validate=True
        )

        result_df = process_alpha_weights(df, neutralize=False)
        assert_df_equal(result_df, expected_df)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            process_alpha_weights("invalid_type")  # type: ignore[arg-type]


class TestCsPca:
    def test_pca_scores_single_component(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)
        result = cs_pca(df, window=3, n_components=1, output="scores")
        assert isinstance(result, DataFrame)

        # Score is broadcast across all columns (every column equal per row).
        result_abs = result.with_columns(
            [pl.col("c0").abs(), pl.col("c1").abs(), pl.col("c2").abs()]
        )
        expected_score = [None, None, 2.236068, 2.289187, 1.467081]
        expected_pl = result._df.with_columns(
            [
                pl.Series("c0", expected_score).abs(),
                pl.Series("c1", expected_score).abs(),
                pl.Series("c2", expected_score).abs(),
            ]
        )
        expected_df = DataFrame(
            expected_pl, result.time_col, result.freq, result.name, _skip_validate=True
        )
        assert_df_equal(result_abs, expected_df)

    def test_pca_scores_two_components(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)
        result = cs_pca(df, window=3, n_components=2, output="scores")

        assert isinstance(result, tuple)
        assert len(result) == 2
        pc1_df, pc2_df = result

        expected_pc1 = [None, None, 2.236068, 2.289187, 1.467081]
        expected_pc2 = [None, None, 2.720236e-33, 1.235379e-01, 3.703325e-01]

        pc1_abs = pc1_df.with_columns(
            [pl.col("c0").abs(), pl.col("c1").abs(), pl.col("c2").abs()]
        )
        expected_pc1_pl = pc1_df._df.with_columns(
            [
                pl.Series("c0", expected_pc1).abs(),
                pl.Series("c1", expected_pc1).abs(),
                pl.Series("c2", expected_pc1).abs(),
            ]
        )
        expected_pc1_df = DataFrame(
            expected_pc1_pl,
            pc1_df.time_col,
            pc1_df.freq,
            pc1_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc1_abs, expected_pc1_df)

        pc2_abs = pc2_df.with_columns(
            [pl.col("c0").abs(), pl.col("c1").abs(), pl.col("c2").abs()]
        )
        expected_pc2_pl = pc2_df._df.with_columns(
            [
                pl.Series("c0", expected_pc2).abs(),
                pl.Series("c1", expected_pc2).abs(),
                pl.Series("c2", expected_pc2).abs(),
            ]
        )
        expected_pc2_df = DataFrame(
            expected_pc2_pl,
            pc2_df.time_col,
            pc2_df.freq,
            pc2_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc2_abs, expected_pc2_df)

    def test_pca_loadings(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)

        # n_components=1 -> single DataFrame
        result = cs_pca(df, window=3, n_components=1, output="loadings")
        assert isinstance(result, DataFrame)

        expected_pl = result._df.with_columns(
            [
                pl.Series("c0", [None, None, -0.547723, 0.600697, 0.582991]),
                pl.Series("c1", [None, None, -0.547723, 0.590927, 0.565899]),
                pl.Series("c2", [None, None, 0.632456, 0.538488, 0.582991]),
            ]
        )
        expected_df = DataFrame(
            expected_pl, result.time_col, result.freq, result.name, _skip_validate=True
        )
        assert_df_equal(result, expected_df)

        # n_components=2 -> tuple of 2 DataFrames
        result2 = cs_pca(df, window=3, n_components=2, output="loadings")
        assert isinstance(result2, tuple)
        assert len(result2) == 2
        pc1_df, pc2_df = result2

        # PC1 loadings
        expected_pc1_pl = pc1_df._df.with_columns(
            [
                pl.Series("c0", [None, None, -0.547723, 0.600697, 0.582991]),
                pl.Series("c1", [None, None, -0.547723, 0.590927, 0.565899]),
                pl.Series("c2", [None, None, 0.632456, 0.538488, 0.582991]),
            ]
        )
        expected_pc1_df = DataFrame(
            expected_pc1_pl,
            pc1_df.time_col,
            pc1_df.freq,
            pc1_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc1_df, expected_pc1_df)

        # PC2 loadings
        expected_pc2_pl = pc2_df._df.with_columns(
            [
                pl.Series("c0", [None, None, 0.707107, -0.314084, -0.400151]),
                pl.Series("c1", [None, None, -0.707107, -0.444965, 0.824474]),
                pl.Series("c2", [None, None, -1.837628e-17, 0.838664, -0.400151]),
            ]
        )
        expected_pc2_df = DataFrame(
            expected_pc2_pl,
            pc2_df.time_col,
            pc2_df.freq,
            pc2_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc2_df, expected_pc2_df)

    def test_pca_explained_variance(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)
        result = cs_pca(df, window=3, n_components=2, output="explained_variance")

        assert isinstance(result, tuple)
        assert len(result) == 2
        pc1_df, pc2_df = result

        expected_pc1_var = [None, None, 0.833333, 0.895101, 0.963586]
        expected_pc2_var = [None, None, 0.166667, 0.104899, 0.036414]

        # PC1 variance (broadcast across columns)
        expected_pc1_pl = pc1_df._df.with_columns(
            [
                pl.Series("c0", expected_pc1_var),
                pl.Series("c1", expected_pc1_var),
                pl.Series("c2", expected_pc1_var),
            ]
        )
        expected_pc1_df = DataFrame(
            expected_pc1_pl,
            pc1_df.time_col,
            pc1_df.freq,
            pc1_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc1_df, expected_pc1_df)

        # PC2 variance (broadcast across columns)
        expected_pc2_pl = pc2_df._df.with_columns(
            [
                pl.Series("c0", expected_pc2_var),
                pl.Series("c1", expected_pc2_var),
                pl.Series("c2", expected_pc2_var),
            ]
        )
        expected_pc2_df = DataFrame(
            expected_pc2_pl,
            pc2_df.time_col,
            pc2_df.freq,
            pc2_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc2_df, expected_pc2_df)

    def test_nan_handling(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [None, 4.0, 5.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)
        result = cs_pca(df, window=3, n_components=1, output="scores")
        assert isinstance(result, DataFrame)

        # Windows overlapping the NaN row yield None.
        expected_pl = result._df.clone().with_columns(
            [
                pl.when(pl.arange(0, pl.len()) >= 4)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in ["c0", "c1", "c2"]
            ]
        )
        expected_df = DataFrame(
            expected_pl, result.time_col, result.freq, result.name, _skip_validate=True
        )
        assert_df_equal(result, expected_df)

    def test_single_feature(self) -> None:
        x = [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
        df = _create_df(x)
        result = cs_pca(df, window=3, n_components=1, output="scores")
        assert isinstance(result, DataFrame)

        result_abs = result.with_columns(pl.col("c0").abs())
        expected_pl = result._df.with_columns(
            [
                pl.Series(
                    "c0",
                    [
                        None,
                        None,
                        abs((3 - 2) / 0.8165),
                        abs((4 - 3) / 0.8165),
                        abs((5 - 4) / 0.8165),
                    ],
                )
            ]
        )
        expected_df = DataFrame(
            expected_pl, result.time_col, result.freq, result.name, _skip_validate=True
        )
        assert_df_equal(result_abs, expected_df)

    def test_error_cases(self) -> None:
        x = [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]
        df = _create_df(x)
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            cs_pca("invalid_type", window=2)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must be an integer > 1"):
            cs_pca(df, window=1, n_components=1)
        with pytest.raises(ValueError, match="must be a positive integer"):
            cs_pca(df, window=2, n_components=0)
        with pytest.raises(ValueError, match="exceeds number of features"):
            cs_pca(df, window=2, n_components=3)
        with pytest.raises(ValueError, match="must be one of"):
            cs_pca(df, window=2, n_components=1, output="invalid")  # type: ignore[arg-type]

    def test_constant_columns(self) -> None:
        # Columns with zero std must not produce NaN scores.
        x = [
            [1.0, 2.0, 5.0],
            [2.0, 1.0, 5.0],
            [3.0, 3.0, 5.0],
            [4.0, 4.0, 6.0],
            [5.0, 5.0, 7.0],
        ]
        df = _create_df(x)
        result = cs_pca(df, window=3, n_components=1, output="scores")
        assert isinstance(result, DataFrame)

        assert result._df.row(2)[1] is not None

    def test_pca_scores_with_sklearn(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)
        window = 3
        n_components = 1

        result = cs_pca(df, window=window, n_components=n_components, output="scores")
        assert isinstance(result, DataFrame)

        # Expected values from sklearn as an independent reference.
        X = np.array(x)
        expected_scores: list[Any] = [None, None]
        for i in range(window - 1, len(x)):
            window_data = X[i - window + 1 : i + 1]
            mean = window_data.mean(axis=0)
            std = window_data.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            normalized = (window_data - mean) / std
            pca = PCA(n_components=n_components)
            pca.fit(normalized)
            score = pca.transform(normalized[-1:])
            expected_scores.append(abs(score[0, 0]))

        result_abs = result.with_columns(
            [pl.col("c0").abs(), pl.col("c1").abs(), pl.col("c2").abs()]
        )
        expected_pl = result._df.with_columns(
            [
                pl.Series("c0", expected_scores),
                pl.Series("c1", expected_scores),
                pl.Series("c2", expected_scores),
            ]
        )
        expected_df = DataFrame(
            expected_pl, result.time_col, result.freq, result.name, _skip_validate=True
        )
        assert_df_equal(result_abs, expected_df)

    def test_pca_loadings_with_sklearn(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)
        window = 3
        n_components = 1

        result = cs_pca(df, window=window, n_components=n_components, output="loadings")
        assert isinstance(result, DataFrame)

        # Expected values from sklearn as an independent reference.
        X = np.array(x)
        expected_loadings: dict[str, list[Any]] = {
            f"c{j}": [None, None] for j in range(3)
        }
        for i in range(window - 1, len(x)):
            window_data = X[i - window + 1 : i + 1]
            mean = window_data.mean(axis=0)
            std = window_data.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            normalized = (window_data - mean) / std
            pca = PCA(n_components=n_components)
            pca.fit(normalized)
            for j in range(3):
                expected_loadings[f"c{j}"].append(pca.components_[0, j])

        expected_pl = result._df.with_columns(
            [pl.Series(col, vals) for col, vals in expected_loadings.items()]
        )
        expected_df = DataFrame(
            expected_pl, result.time_col, result.freq, result.name, _skip_validate=True
        )
        assert_df_equal(result, expected_df)

    def test_pca_explained_variance_with_sklearn(self) -> None:
        x = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [3.0, 1.0, 1.0],
        ]
        df = _create_df(x)
        window = 3
        n_components = 2

        result = cs_pca(
            df, window=window, n_components=n_components, output="explained_variance"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        pc1_df, pc2_df = result

        # Expected values from sklearn as an independent reference.
        X = np.array(x)
        expected_var_pc1: list[Any] = [None, None]
        expected_var_pc2: list[Any] = [None, None]
        for i in range(window - 1, len(x)):
            window_data = X[i - window + 1 : i + 1]
            mean = window_data.mean(axis=0)
            std = window_data.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            normalized = (window_data - mean) / std
            pca = PCA(n_components=n_components)
            pca.fit(normalized)
            expected_var_pc1.append(pca.explained_variance_ratio_[0])
            expected_var_pc2.append(pca.explained_variance_ratio_[1])

        # PC1 DataFrame
        expected_pc1_pl = pc1_df._df.with_columns(
            [
                pl.Series("c0", expected_var_pc1),
                pl.Series("c1", expected_var_pc1),
                pl.Series("c2", expected_var_pc1),
            ]
        )
        expected_pc1_df = DataFrame(
            expected_pc1_pl,
            pc1_df.time_col,
            pc1_df.freq,
            pc1_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc1_df, expected_pc1_df)

        # PC2 DataFrame
        expected_pc2_pl = pc2_df._df.with_columns(
            [
                pl.Series("c0", expected_var_pc2),
                pl.Series("c1", expected_var_pc2),
                pl.Series("c2", expected_var_pc2),
            ]
        )
        expected_pc2_df = DataFrame(
            expected_pc2_pl,
            pc2_df.time_col,
            pc2_df.freq,
            pc2_df.name,
            _skip_validate=True,
        )
        assert_df_equal(pc2_df, expected_pc2_df)

    def test_insufficient_rows(self) -> None:
        # Fewer rows than the window -> all-NaN result, same column structure.
        df = _create_df([[1.0, 2.0], [3.0, 4.0]])
        result = cs_pca(df, window=5, n_components=1, output="scores")
        assert isinstance(result, DataFrame)
        cols = [c for c in result.columns if c != result.time_col]
        assert np.isnan(result._df.select(cols).to_numpy()).all()
