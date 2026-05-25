from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from ctalearn.core.dataframe import DataFrame
from ctalearn.operator._arithmetic import (
    cbrt,
    identity,
    log,
    sign,
    sqrt,
    symmetric_log,
    symmetric_sqrt,
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


class TestSign:
    def test_basic(self) -> None:
        input_df = _create_df({"col1": [1.0, 2.0, -4.0, -3.0, 0.0, 2.0, None]})
        result = sign(input_df)

        expected_df = _create_df(
            {"col1": [1.0, 1.0, -1.0, -1.0, 0.0, 1.0, None]}, name="sign(df)"
        )

        assert_df_equal(result, expected_df)

    def test_multi_dims(self) -> None:
        input_df = _create_df(
            {
                "c1": [1.0, 2.0, 3.0, 5.0, 3.0, None],
                "c2": [2.0, 1.0, -3.0, 0.0, 1.0, 4.0],
                "c3": [-3.0, -3.0, 1.0, -7.0, -1.0, -5.0],
            }
        )
        result = sign(input_df)

        expected_df = _create_df(
            {
                "c1": [1.0, 1.0, 1.0, 1.0, 1.0, None],
                "c2": [1.0, 1.0, -1.0, 0.0, 1.0, 1.0],
                "c3": [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
            },
            name="sign(df)",
        )
        assert_df_equal(result, expected_df)

    def test_nan(self) -> None:
        # NaN input -> null (consistent with null handling).
        df = _create_df({"val": [1.0, float("nan"), -2.0]})
        result = sign(df)
        expected = _create_df({"val": [1.0, None, -1.0]}, name="sign(df)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            sign("invalid_type")  # type: ignore[arg-type]


class TestLog:
    def test_basic(self) -> None:
        # Cases: Positive, e, 1, 0, Negative, NaN
        val = [np.e, 1, 0.5, 0, -5, None]
        df = _create_df({"val": val})

        result = log(df)

        expected_val = [1.0, 0.0, np.log(0.5), None, None, None]
        expected = _create_df({"val": expected_val}, name="log(df)")

        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            log("invalid_type")  # type: ignore[arg-type]

    def test_nan(self) -> None:
        # Out-of-domain (<=0) and NaN input both -> null.
        df = _create_df({"val": [np.e, float("nan"), -1.0]})
        result = log(df)
        expected = _create_df({"val": [1.0, None, None]}, name="log(df)")
        assert_df_equal(result, expected)


class TestSymmetricLog:
    def test_basic(self) -> None:
        # Cases: Positive, Negative, Zero
        # Formula: sign(x) * log(|x| + 1)
        val = [10.0, -10.0, 0.0]
        df = _create_df({"val": val})

        result = symmetric_log(df)

        v10 = np.log(11)
        expected_val = [v10, -v10, 0.0]
        expected = _create_df({"val": expected_val}, name="symmetric_log(df)")

        assert_df_equal(result, expected)

    def test_nan(self) -> None:
        df = _create_df({"val": [10.0, float("nan")]})
        result = symmetric_log(df)
        expected = _create_df({"val": [np.log(11), None]}, name="symmetric_log(df)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            symmetric_log("invalid_type")  # type: ignore[arg-type]


class TestSqrt:
    def test_basic(self) -> None:
        # Cases: Perfect square, Zero, Negative
        val = [4.0, 0.0, -4.0, 2.0]
        df = _create_df({"val": val})

        result = sqrt(df)

        expected_val = [2.0, 0.0, None, np.sqrt(2)]
        expected = _create_df({"val": expected_val}, name="sqrt(df)")

        assert_df_equal(result, expected)

    def test_nan(self) -> None:
        # Out-of-domain (<0) and NaN input both -> null.
        df = _create_df({"val": [4.0, float("nan"), -4.0]})
        result = sqrt(df)
        expected = _create_df({"val": [2.0, None, None]}, name="sqrt(df)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            sqrt("invalid_type")  # type: ignore[arg-type]


class TestSymmetricSqrt:
    def test_basic(self) -> None:
        # Cases: Positive, Negative, Zero
        # Formula: sign(x) * sqrt(|x|)
        val = [4.0, -4.0, 0.0]
        df = _create_df({"val": val})

        result = symmetric_sqrt(df)

        expected_val = [2.0, -2.0, 0.0]
        expected = _create_df({"val": expected_val}, name="symmetric_sqrt(df)")

        assert_df_equal(result, expected)

    def test_multi_col(self) -> None:
        # Test broadcasting
        df = _create_df({"c1": [4.0, -4.0], "c2": [100.0, -100.0]})
        result = symmetric_sqrt(df)

        expected = _create_df(
            {"c1": [2.0, -2.0], "c2": [10.0, -10.0]}, name="symmetric_sqrt(df)"
        )
        assert_df_equal(result, expected)

    def test_nan(self) -> None:
        df = _create_df({"val": [4.0, float("nan")]})
        result = symmetric_sqrt(df)
        expected = _create_df({"val": [2.0, None]}, name="symmetric_sqrt(df)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            symmetric_sqrt("invalid_type")  # type: ignore[arg-type]


class TestCbrt:
    def test_basic(self) -> None:
        df = _create_df({"val": [8.0, 0.0, -27.0, 1.0, -1.0]})
        result = cbrt(df)
        expected = _create_df({"val": [2.0, 0.0, -3.0, 1.0, -1.0]}, name="cbrt(df)")
        assert_df_equal(result, expected)

    def test_nan(self) -> None:
        df = _create_df({"val": [8.0, float("nan")]})
        result = cbrt(df)
        expected = _create_df({"val": [2.0, None]}, name="cbrt(df)")
        assert_df_equal(result, expected)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            cbrt("invalid_type")  # type: ignore[arg-type]


class TestIdentity:
    def test_basic(self) -> None:
        df = _create_df({"val": [1.0, -2.0, 3.0]})
        result = identity(df)
        assert_df_equal(result, df)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Type of `df` should be DataFrame"):
            identity("invalid_type")  # type: ignore[arg-type]
