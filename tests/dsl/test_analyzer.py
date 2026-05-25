from typing import Any

import pytest
from lark.exceptions import VisitError

from ctalearn.dsl import (
    Arg,
    DslType,
    TypeCheckTransformer,
    parser,
)
from ctalearn.dsl.exceptions import DslTypeError
from tests.dsl.fixtures import schema_env


class TestAnalyzer:
    """Test suite for static type checking (TypeCheckTransformer)."""

    def test_success(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """Ensure a valid AST passes type checking and returns the correct type."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        code = """
            vol = ts_zscore(close, 40 / 2);
            signal = vol * -1 + 4 / 2;
            return cs_rank(signal);
        """
        tree = parser.parse(code)
        result_type = checker.transform(tree)

        assert result_type == DslType.DATAFRAME

    def test_incorrect_return_type(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """Ensure an error is raised if the final returned type is not a DataFrame."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        code = """
            vol = ts_zscore(close, 20);
            return 5; # Should raise error
        """
        tree = parser.parse(code)

        with pytest.raises(VisitError) as exc_info:
            checker.transform(tree)

        assert isinstance(exc_info.value.orig_exc, DslTypeError)
        assert "Return type should be DataFrame" in str(exc_info.value.orig_exc)

    def test_function_argument_type_mismatch(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """Ensure an error is raised if function arguments have incorrect types."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        # ts_zscore expects (DATAFRAME, INT), but gets (DATAFRAME, DATAFRAME)
        code = "return ts_zscore(close, volume);"
        tree = parser.parse(code)

        with pytest.raises(VisitError) as exc_info:
            checker.transform(tree)

        assert isinstance(exc_info.value.orig_exc, DslTypeError)
        assert "2-th argument should be int" in str(exc_info.value.orig_exc)

    def test_undefined_variable(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """Ensure an error is raised when an undefined variable is referenced."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        code = "return invalid_factor;"
        tree = parser.parse(code)

        with pytest.raises(VisitError) as exc_info:
            checker.transform(tree)

        assert isinstance(exc_info.value.orig_exc, DslTypeError)
        assert "Unknown variable: 'invalid_factor'" in str(exc_info.value.orig_exc)

    def test_undefined_function(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """Ensure an error is raised when an undefined function is called."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        code = """
            a = invalid_function(close, volume, 20);
            return a;
        """
        tree = parser.parse(code)

        with pytest.raises(VisitError) as exc_info:
            checker.transform(tree)

        assert isinstance(exc_info.value.orig_exc, DslTypeError)
        assert "Unknown function: 'invalid_function'" in str(exc_info.value.orig_exc)

    def test_subtraction_typecheck(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """Subtraction resolves its operand types (the `sub` rule)."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        tree = parser.parse("return cs_rank(close - open);")
        assert checker.transform(tree) == DslType.DATAFRAME

    def test_float_scalar_promotes(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """A float-with-int scalar op (no DataFrame) resolves to FLOAT."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        # x = 1.0 + 2 exercises resolve_binop_type(FLOAT, INT) -> FLOAT.
        code = """
            x = 1.0 + 2;
            return cs_rank(close);
        """
        assert checker.transform(parser.parse(code)) == DslType.DATAFRAME

    def test_wrong_argument_count(
        self, schema_env: tuple[dict[str, DslType], dict[str, dict[str, Any]]]
    ) -> None:
        """All-required function called with too many args reports an exact count."""
        factor_schema, func_schema = schema_env
        checker = TypeCheckTransformer(factor_schema, func_schema)

        # cs_rank expects exactly 1 argument.
        with pytest.raises(VisitError) as exc_info:
            checker.transform(parser.parse("return cs_rank(close, volume);"))

        assert isinstance(exc_info.value.orig_exc, DslTypeError)
        assert "expect 1 arguments, got 2" in str(exc_info.value.orig_exc)

    def test_optional_float_arg_accepts_int(self) -> None:
        """A FLOAT parameter accepts an INT argument; optional arg may be supplied."""
        factor_schema = {"close": DslType.DATAFRAME}
        func_schema = {
            "scale": {
                "args": [Arg(DslType.DATAFRAME), Arg(DslType.FLOAT, default=0.0)],
                "return": DslType.DATAFRAME,
            }
        }
        checker = TypeCheckTransformer(factor_schema, func_schema)

        # INT literal 5 supplied where a FLOAT is expected -> accepted.
        tree = parser.parse("return scale(close, 5);")
        assert checker.transform(tree) == DslType.DATAFRAME

    def test_optional_arg_too_many(self) -> None:
        """Function with an optional arg reports a count range when over-supplied."""
        factor_schema = {"close": DslType.DATAFRAME}
        func_schema = {
            "scale": {
                "args": [Arg(DslType.DATAFRAME), Arg(DslType.FLOAT, default=0.0)],
                "return": DslType.DATAFRAME,
            }
        }
        checker = TypeCheckTransformer(factor_schema, func_schema)

        # 3 args supplied; valid range is 1-2.
        with pytest.raises(VisitError) as exc_info:
            checker.transform(parser.parse("return scale(close, 1, 2);"))

        assert isinstance(exc_info.value.orig_exc, DslTypeError)
        assert "expect 1-2 arguments, got 3" in str(exc_info.value.orig_exc)
