import pytest

from ctalearn.dsl import (
    parser,
)


class TestParser:
    """Test suite for the Lark grammar and lexing rules."""

    def test_missing_semicolon(self) -> None:
        """Ensure the parser raises an error if a statement lacks a semicolon."""
        code = "a = close \n return a;"
        with pytest.raises(Exception):
            parser.parse(code)

    def test_missing_return(self) -> None:
        """Ensure the parser raises an error if the script lacks a return statement."""
        code = "a = close;"
        with pytest.raises(Exception):
            parser.parse(code)

    def test_empty_expression(self) -> None:
        """Ensure the parser correctly handles empty statements."""
        code = ";;; return close;"
        parser.parse(code)

    def test_comment(self) -> None:
        """Ensure the parser correctly ignores inline and full-line comments."""
        code = """
            vol = ts_zscore(close, 10);  # This is an example comment
            return vol;                  # Another example comment
        """
        parser.parse(code)

    def test_unassigned_expression(self) -> None:
        """Ensure the parser rejects unassigned function calls or expressions."""
        # Function executed but not assigned to a variable or returned.
        code = """
            ts_zscore(close, 10);
            return close;
        """
        # Lark will raise an UnexpectedToken or UnexpectedCharacters exception.
        with pytest.raises(Exception):
            parser.parse(code)

    def test_decimal_numbers(self) -> None:
        """Integers and decimal floats are accepted."""
        parser.parse("return close * 100000;")
        parser.parse("return close * 1.5;")
        parser.parse("return close * 0.00000001;")

    def test_scientific_notation_rejected(self) -> None:
        """Exponent literals are not supported; reject them at parse time."""
        with pytest.raises(Exception):
            parser.parse("return close * 1e5;")
        with pytest.raises(Exception):
            parser.parse("return close * 1e-8;")
