from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest
from lark.exceptions import VisitError

from ctalearn.core.dataframe import DataFrame
from ctalearn.dsl import (
    Arg,
    DslType,
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

    def test_overload_dispatch_by_runtime_type(self) -> None:
        """Multi-overload runtime: dispatch picks the callable by actual arg type."""
        df_calls: list[Any] = []
        scalar_calls: list[Any] = []

        def df_sqrt(x: Any) -> Any:
            df_calls.append(x)
            return _make_df([1.0])

        def scalar_sqrt(x: Any) -> Any:
            scalar_calls.append(x)
            return _make_df([float(x) ** 0.5])

        functions = {
            "sqrt": [
                (df_sqrt, [Arg(DslType.DATAFRAME)]),
                (scalar_sqrt, [Arg(DslType.FLOAT)]),
            ],
            # cs_rank wraps the scalar branch's result so the return is a DataFrame.
            "cs_rank": [
                (lambda v: v, [Arg(DslType.DATAFRAME)]),
            ],
        }
        data_loaders = {"close": lambda: _make_df([4.0])}
        interp = ExecutionTransformer(functions, data_loaders)

        # DataFrame arg -> df_sqrt
        interp.transform(parser.parse("return sqrt(close);"))
        assert len(df_calls) == 1 and len(scalar_calls) == 0

        # float arg -> scalar_sqrt (wrap in cs_rank to satisfy return-type rule)
        ExecutionTransformer(functions, data_loaders).transform(
            parser.parse("x = sqrt(9.0); return cs_rank(close);")
        )
        assert len(scalar_calls) == 1 and scalar_calls[0] == 9.0

    def test_overload_dispatch_no_match_at_runtime(self) -> None:
        """Defensive: when host skips analyzer and no overload fits, runtime raises."""
        functions = {
            # Neither overload accepts a FLOAT (INT doesn't widen to anything here).
            "weird": [
                (lambda x: x, [Arg(DslType.INT)]),
                (lambda x: x, [Arg(DslType.DATAFRAME)]),
            ],
        }
        interp = ExecutionTransformer(functions, {})

        with pytest.raises(VisitError) as exc_info:
            interp.transform(parser.parse("return weird(1.5);"))

        assert isinstance(exc_info.value.orig_exc, DslRuntimeError)
        assert "No matching overload for 'weird'" in str(exc_info.value.orig_exc)

    def test_py_to_dsl_int_path(self) -> None:
        """INT runtime arg maps to DslType.INT (covers the int branch in _py_to_dsl)."""
        from ctalearn.dsl.interpreter import _py_to_dsl

        assert _py_to_dsl(5) == DslType.INT

    def test_py_to_dsl_rejects_bool(self) -> None:
        """bool reaching runtime dispatch is a host bug (DSL has no bool literal)."""
        from ctalearn.dsl.interpreter import _py_to_dsl

        with pytest.raises(DslRuntimeError, match="bool is not a DSL value type"):
            _py_to_dsl(True)

    def test_py_to_dsl_rejects_unknown_type(self) -> None:
        """A runtime value with no DslType counterpart is rejected."""
        from ctalearn.dsl.interpreter import _py_to_dsl

        with pytest.raises(DslRuntimeError, match="Unsupported runtime type"):
            _py_to_dsl(["not a dsl value"])

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

        functions["cs_rank"] = [(raising, [Arg(DslType.DATAFRAME)])]
        interpreter = ExecutionTransformer(functions, data_loaders)

        with pytest.raises(VisitError) as exc_info:
            interpreter.transform(parser.parse("return cs_rank(close);"))

        assert isinstance(exc_info.value.orig_exc, DslRuntimeError)
        assert "Failed to execution function 'cs_rank'" in str(exc_info.value.orig_exc)

    def test_string_args_passed_unquoted(self) -> None:
        """Functions receive string literals/variables as `str`, quotes stripped."""
        received: list[Any] = []

        def f(df: Any, s: Any) -> Any:
            received.append(s)
            return df

        functions = {"f": [(f, [Arg(DslType.DATAFRAME), Arg(DslType.STRING)])]}
        data_loaders = {"close": lambda: _make_df([1.0])}
        code = """
            s = "scores";
            x = f(close, "");
            y = f(x, "a#b");
            return f(y, s);
        """
        ExecutionTransformer(functions, data_loaders).transform(parser.parse(code))

        assert received == ["", "a#b", "scores"]

    def test_overload_dispatch_on_string(self) -> None:
        """Multi-overload runtime: a `str` arg dispatches to the STRING overload."""
        calls: list[tuple[str, Any]] = []

        def by_int(df: Any, n: Any) -> Any:
            calls.append(("int", n))
            return df

        def by_str(df: Any, s: Any) -> Any:
            calls.append(("str", s))
            return df

        functions = {
            "f": [
                (by_int, [Arg(DslType.DATAFRAME), Arg(DslType.INT)]),
                (by_str, [Arg(DslType.DATAFRAME), Arg(DslType.STRING)]),
            ]
        }
        data_loaders = {"close": lambda: _make_df([1.0])}
        code = 'x = f(close, 3); return f(x, "3");'
        ExecutionTransformer(functions, data_loaders).transform(parser.parse(code))

        assert calls == [("int", 3), ("str", "3")]

    def test_data_loader_string(self) -> None:
        """A host loader may return a `str`, usable as a STRING argument.

        Two overloads force runtime dispatch, so the loaded `str` must map to STRING.
        """
        received: list[Any] = []

        def f(df: Any, s: Any) -> Any:
            received.append(s)
            return df

        functions = {
            "f": [
                (f, [Arg(DslType.DATAFRAME), Arg(DslType.STRING)]),
                (f, [Arg(DslType.DATAFRAME), Arg(DslType.INT)]),
            ]
        }
        data_loaders = {"close": lambda: _make_df([1.0]), "mode": lambda: "scores"}
        ExecutionTransformer(functions, data_loaders).transform(
            parser.parse("return f(close, mode);")
        )

        assert received == ["scores"]
