import functools
from typing import Any

import pytest

from ctalearn.dsl.schema import Arg, DslType


@pytest.fixture
def schema_env() -> tuple[dict[str, DslType], dict[str, dict[str, Any]]]:
    """Provides factor and function schemas for static analysis.

    Returns:
        A tuple containing the factor schema and the function schema dictionaries.
    """
    factor_schema = {
        "open": DslType.DATAFRAME,
        "high": DslType.DATAFRAME,
        "low": DslType.DATAFRAME,
        "close": DslType.DATAFRAME,
        "volume": DslType.DATAFRAME,
    }
    function_schema = {
        "ts_zscore": {
            "args": [Arg(DslType.DATAFRAME), Arg(DslType.INT)],
            "return": DslType.DATAFRAME,
        },
        "cs_rank": {
            "args": [Arg(DslType.DATAFRAME)],
            "return": DslType.DATAFRAME,
        },
    }
    return factor_schema, function_schema


@pytest.fixture
def runtime_env() -> tuple[
    dict[str, Any],
    dict[str, Any],
    set[str],
]:
    """Provides mock functions and lazy-loading data fetchers.

    Returns:
        A tuple containing the data loaders, mathematical functions,
        and a set tracking fetched data.
    """
    # Mock data store
    fetched_data: set[str] = set()

    def mock_fetch(factor: str) -> str:
        fetched_data.add(factor)
        return f"DF_{factor.upper()}_DATA"

    data_loaders = {
        factor_str: functools.partial(mock_fetch, factor_str)
        for factor_str in ["open", "high", "low", "close", "volume"]
    }

    # Mock mathematical functions
    functions = {
        "ts_zscore": lambda df, window: f"ZSCORE({df}, {window})",
        "cs_rank": lambda df: f"RANK({df})",
    }

    return data_loaders, functions, fetched_data
