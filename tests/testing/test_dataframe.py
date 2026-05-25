from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from ctalearn.core.dataframe import DataFrame
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


class TestAssertDfEqual:
    def test_equal_passes(self) -> None:
        a = _create_df({"x": [1.0, 2.0, 3.0]})
        b = _create_df({"x": [1.0, 2.0, 3.0]})
        assert_df_equal(a, b)  # no raise

    def test_column_order_ignored(self) -> None:
        # The whole reason this helper exists: order must not matter.
        a = _create_df({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        b = DataFrame(a._df.select(["y", "x", "ts"]), "ts", freq=a.freq, name=a.name)
        assert_df_equal(a, b)

    def test_value_mismatch_raises(self) -> None:
        a = _create_df({"x": [1.0, 2.0, 3.0]})
        b = _create_df({"x": [1.0, 2.0, 9.0]})
        with pytest.raises(AssertionError):
            assert_df_equal(a, b)

    def test_name_mismatch_raises(self) -> None:
        a = _create_df({"x": [1.0, 2.0]}, name="alpha")
        b = _create_df({"x": [1.0, 2.0]}, name="beta")
        with pytest.raises(AssertionError, match="name mismatch"):
            assert_df_equal(a, b)

    def test_time_col_mismatch_raises(self) -> None:
        a = _create_df({"x": [1.0, 2.0]})
        b = _create_df({"x": [1.0, 2.0]})
        b.time_col = "other"  # frames stay value-equal; only metadata differs
        with pytest.raises(AssertionError, match="time_col mismatch"):
            assert_df_equal(a, b)

    def test_freq_mismatch_raises(self) -> None:
        a = _create_df({"x": [1.0, 2.0]})
        b = _create_df({"x": [1.0, 2.0]})
        b.freq = a.freq + 1  # type: ignore[operator]
        with pytest.raises(AssertionError, match="freq mismatch"):
            assert_df_equal(a, b)
