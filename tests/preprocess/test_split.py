from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from ctalearn.core.dataframe import DataFrame
from ctalearn.preprocess import train_test_split


def _create_df(data: dict[str, Any], name: str = "df") -> DataFrame:
    length = len(data[next(iter(data))])
    dates = pl.datetime_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    return DataFrame(pl.DataFrame({**data, "ts": dates}), "ts", name=name)


class TestTrainTestSplit:
    def test_basic(self) -> None:
        df = _create_df({"a": np.arange(10)})
        train_df, test_df = train_test_split(df, train_size=0.8)

        assert len(train_df._df) == 8
        assert len(test_df._df) == 2
        assert len(train_df._df) + len(test_df._df) == len(df._df)
        assert train_df.name == test_df.name == df.name

    def test_invalid_train_size(self) -> None:
        df = _create_df({"a": np.arange(10)})

        with pytest.raises(ValueError):
            train_test_split(df, train_size=1.5)

        with pytest.raises(ValueError):
            train_test_split(df, train_size=-0.5)

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            train_test_split("not_a_dataframe", train_size=0.5)  # type: ignore[arg-type]
