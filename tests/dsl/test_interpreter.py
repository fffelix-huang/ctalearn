from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest
from lark.exceptions import VisitError

from ctalearn.core.dataframe import DataFrame
from ctalearn.dsl import (
    ExecutionTransformer,
    parser,
)
from ctalearn.dsl.exceptions import DslRuntimeError
from tests.dsl.fixtures import runtime_env


def _make_df(values: list[float]) -> DataFrame:
    """A single-series DataFrame for exercising DSL arithmetic on real data."""
    ts = pl.datetime_range(
        datetime(2023, 1, 1),
        datetime(2023, 1, 1) + timedelta(seconds=len(values) - 1),
        interval="1s",
        eager=True,
    )
    return DataFrame(pl.DataFrame({"v": values, "ts": ts}), "ts")


class TestInterpreter:
    """Test suite for runtime execution and lazy loading (ExecutionTransformer)."""

    def test_success_and_lazy_loading(
        self, runtime_env: tuple[dict[str, Any], dict[str, Any], set[str]]
    ) -> None:
        """Ensure AST executes correctly and with lazy loading."""
        data_loaders, functions, fetched_data = runtime_env
        interpreter = ExecutionTransformer(functions, data_loaders)

        code = """
            # Only 'close' is used, 'volume' should NOT be fetched
            vol = ts_zscore(close, 20);
            return cs_rank(vol);
        """
        tree = parser.parse(code)
        result = interpreter.transform(tree)

        assert result == "RANK(ZSCORE(DF_CLOSE_DATA, 20))"

        # Verify Lazy Loading works: "close" was fetched, but "volume" was not
        assert "close" in fetched_data
        assert "volume" not in fetched_data

    def test_missing_data_loader(
        self, runtime_env: tuple[dict[str, Any], dict[str, Any], set[str]]
    ) -> None:
        """Ensure runtime error is raised if a required data loader is missing."""
        data_loaders, functions, _ = runtime_env

        # Remove 'close' from available loaders to simulate a configuration error
        del data_loaders["close"]
        interpreter = ExecutionTransformer(functions, data_loaders)

        code = "return ts_zscore(close, 20);"
        tree = parser.parse(code)

        with pytest.raises(VisitError) as exc_info:
            interpreter.transform(tree)

        assert isinstance(exc_info.value.orig_exc, DslRuntimeError)
        assert "Unknown variable 'close'" in str(exc_info.value.orig_exc)

    def test_unregistered_function_runtime_error(
        self, runtime_env: tuple[dict[str, Any], dict[str, Any], set[str]]
    ) -> None:
        """A function missing from the runtime registry raises DslRuntimeError."""
        data_loaders, functions, _ = runtime_env
        del functions["cs_rank"]
        interpreter = ExecutionTransformer(functions, data_loaders)

        tree = parser.parse("return cs_rank(close);")
        with pytest.raises(VisitError) as exc_info:
            interpreter.transform(tree)

        assert isinstance(exc_info.value.orig_exc, DslRuntimeError)

    def test_arithmetic_operators(self) -> None:
        """+, -, *, / and unary - execute on DataFrame operands (the DSL's type)."""
        data_loaders = {
            "close": lambda: _make_df([4.0, 6.0, 8.0]),
            "open": lambda: _make_df([1.0, 2.0, 4.0]),
        }

        def run(code: str) -> Any:
            return ExecutionTransformer({}, data_loaders).transform(parser.parse(code))

        assert run("return close + open;")._df["v"].to_list() == [5.0, 8.0, 12.0]
        assert run("return close - open;")._df["v"].to_list() == [3.0, 4.0, 4.0]
        assert run("return close * open;")._df["v"].to_list() == [4.0, 12.0, 32.0]
        assert run("return close / open;")._df["v"].to_list() == [4.0, 3.0, 2.0]
        assert run("return -close;")._df["v"].to_list() == [-4.0, -6.0, -8.0]

    def test_data_loader_failure(
        self, runtime_env: tuple[dict[str, Any], dict[str, Any], set[str]]
    ) -> None:
        """A loader that raises is wrapped in DslRuntimeError."""
        data_loaders, functions, _ = runtime_env

        def boom() -> str:
            raise RuntimeError("network down")

        data_loaders["close"] = boom
        interpreter = ExecutionTransformer(functions, data_loaders)

        with pytest.raises(VisitError) as exc_info:
            interpreter.transform(parser.parse("return cs_rank(close);"))

        assert isinstance(exc_info.value.orig_exc, DslRuntimeError)
        assert "Failed to fetch 'close'" in str(exc_info.value.orig_exc)

    def test_data_loader_returns_none(
        self, runtime_env: tuple[dict[str, Any], dict[str, Any], set[str]]
    ) -> None:
        """A loader returning None is a runtime error."""
        data_loaders, functions, _ = runtime_env
        data_loaders["close"] = lambda: None
        interpreter = ExecutionTransformer(functions, data_loaders)

        with pytest.raises(VisitError) as exc_info:
            interpreter.transform(parser.parse("return cs_rank(close);"))

        assert isinstance(exc_info.value.orig_exc, DslRuntimeError)
        assert "is None after fetching" in str(exc_info.value.orig_exc)

    def test_function_execution_failure(
        self, runtime_env: tuple[dict[str, Any], dict[str, Any], set[str]]
    ) -> None:
        """An exception inside a registered function surfaces as DslRuntimeError."""
        data_loaders, functions, _ = runtime_env

        def raising(df: Any) -> Any:
            raise ValueError("boom")

        functions["cs_rank"] = raising
        interpreter = ExecutionTransformer(functions, data_loaders)

        with pytest.raises(VisitError) as exc_info:
            interpreter.transform(parser.parse("return cs_rank(close);"))

        assert isinstance(exc_info.value.orig_exc, DslRuntimeError)
        assert "Failed to execution function 'cs_rank'" in str(exc_info.value.orig_exc)
