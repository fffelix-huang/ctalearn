"""End-to-end example: running an alpha expression through the ctalearn DSL.

This is the exact integration pattern a consuming service follows. The service
supplies only two project-specific things:

  * ``factor_schema``  — factor name -> DslType (for static type checking)
  * ``data_loaders``   — factor name -> zero-arg callable returning a DataFrame
                         (called lazily, on first reference)

Everything on the function side (the operators and their signatures) comes from
the library via ``BUILTIN_FUNCTIONS`` / ``BUILTIN_FUNCTION_SCHEMA``.

Run it directly:  ``python examples/dsl_interpreter.py``
"""

from datetime import datetime, timedelta

import polars as pl

from ctalearn.core.dataframe import DataFrame
from ctalearn.dsl import (
    BUILTIN_FUNCTION_SCHEMA,
    BUILTIN_FUNCTIONS,
    DslType,
    ExecutionTransformer,
    TypeCheckTransformer,
    parser,
)


def _make_factor(values: dict[str, list[float]], name: str) -> DataFrame:
    """Build a wide-panel DataFrame: a time column plus one column per asset."""
    length = len(next(iter(values.values())))
    ts = pl.datetime_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    return DataFrame(pl.DataFrame({**values, "ts": ts}), "ts", name=name)


def main() -> None:
    # 1. Project-supplied factors. Each loader is lazy: only factors actually
    #    referenced by the expression are built.
    high = _make_factor(
        {"AAA": [1.0, 2.0, 3.0, 4.0, 5.0], "BBB": [6.0, 7.0, 8.0, 9.0, 10.0]}, "high"
    )
    low = _make_factor(
        {"AAA": [1.0, 1.0, 1.5, 2.0, 2.0], "BBB": [2.0, 1.0, 0.5, 1.0, 2.0]}, "low"
    )
    data_loaders = {"high": lambda: high, "low": lambda: low}
    factor_schema = {"high": DslType.DATAFRAME, "low": DslType.DATAFRAME}

    # 2. The alpha expression.
    code = """
        return ts_mean(high, 3) - low;  # This is a comment
    """

    # 3. Parse -> static type check -> execute.
    tree = parser.parse(code)
    TypeCheckTransformer(factor_schema, BUILTIN_FUNCTION_SCHEMA).transform(tree)
    result = ExecutionTransformer(BUILTIN_FUNCTIONS, data_loaders).transform(tree)

    print(f"{len(BUILTIN_FUNCTIONS)} builtins available; expression evaluated to:")
    print(result)


if __name__ == "__main__":
    main()
