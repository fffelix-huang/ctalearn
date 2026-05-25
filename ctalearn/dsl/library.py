"""Built-in DSL function registry.

Single source of truth binding every exposed ctalearn operator to both its runtime
callable (for `ExecutionTransformer`) and its type signature (for
`TypeCheckTransformer`). Consuming projects supply only `factor_schema` and
`data_loaders`; the function side comes from here.

Conventions:
- `bool` parameters are omitted from the schema entirely. The grammar has no bool
  literal, so they can never be supplied from the DSL; the operator's own Python
  default applies at call time.
- Operators the DSL cannot express are omitted from the registry, e.g. `cs_pca`
  (`Literal` `output` arg + tuple return) and `regression_neut` (list of factor
  DataFrames).
- Optional args must be trailing (positional grammar); `_build` enforces this.
"""

from collections.abc import Callable
from typing import Any

from ctalearn.dsl.schema import Arg, DslType
from ctalearn.operator import (
    cbrt,
    cs_mean,
    cs_rank,
    cs_winsorize,
    cs_zscore,
    identity,
    log,
    sign,
    sqrt,
    symmetric_log,
    symmetric_sqrt,
    ts_corr,
    ts_decay_linear,
    ts_delay,
    ts_delta,
    ts_ffill,
    ts_hurst_exponent,
    ts_max,
    ts_mean,
    ts_median,
    ts_min,
    ts_rank,
    ts_robust_zscore,
    ts_scale,
    ts_std_dev,
    ts_sum,
    ts_zscore,
    vector_neut,
)

DF = DslType.DATAFRAME
INT = DslType.INT
FLOAT = DslType.FLOAT

# name -> (callable, [Arg, ...], return_type)
_Spec = tuple[Callable[..., Any], list[Arg], DslType]

_SPECS: dict[str, _Spec] = {
    # --- arithmetic (element-wise) ---
    "sign": (sign, [Arg(DF)], DF),
    "log": (log, [Arg(DF)], DF),
    "symmetric_log": (symmetric_log, [Arg(DF)], DF),
    "sqrt": (sqrt, [Arg(DF)], DF),
    "symmetric_sqrt": (symmetric_sqrt, [Arg(DF)], DF),
    "cbrt": (cbrt, [Arg(DF)], DF),
    "identity": (identity, [Arg(DF)], DF),
    # --- cross-sectional (ignore_nan: bool omitted) ---
    "cs_mean": (cs_mean, [Arg(DF)], DF),
    "cs_rank": (cs_rank, [Arg(DF)], DF),
    "cs_zscore": (cs_zscore, [Arg(DF)], DF),
    "cs_winsorize": (cs_winsorize, [Arg(DF), Arg(FLOAT)], DF),
    "vector_neut": (vector_neut, [Arg(DF), Arg(DF)], DF),
    # --- time-series ---
    "ts_rank": (ts_rank, [Arg(DF), Arg(INT), Arg(FLOAT, default=0.0)], DF),
    "ts_mean": (ts_mean, [Arg(DF), Arg(INT)], DF),
    "ts_median": (ts_median, [Arg(DF), Arg(INT)], DF),
    "ts_std_dev": (ts_std_dev, [Arg(DF), Arg(INT), Arg(INT, default=0)], DF),
    "ts_zscore": (ts_zscore, [Arg(DF), Arg(INT)], DF),
    "ts_robust_zscore": (ts_robust_zscore, [Arg(DF), Arg(INT)], DF),
    "ts_sum": (ts_sum, [Arg(DF), Arg(INT)], DF),
    "ts_min": (ts_min, [Arg(DF), Arg(INT)], DF),
    "ts_max": (ts_max, [Arg(DF), Arg(INT)], DF),
    "ts_scale": (ts_scale, [Arg(DF), Arg(INT), Arg(FLOAT, default=0.0)], DF),
    "ts_decay_linear": (
        ts_decay_linear,
        [Arg(DF), Arg(INT)],
        DF,
    ),  # dense: bool omitted
    "ts_delay": (ts_delay, [Arg(DF), Arg(INT)], DF),
    "ts_delta": (ts_delta, [Arg(DF), Arg(INT)], DF),
    "ts_ffill": (ts_ffill, [Arg(DF), Arg(INT, default=None)], DF),
    "ts_corr": (ts_corr, [Arg(DF), Arg(DF), Arg(INT)], DF),
    "ts_hurst_exponent": (
        ts_hurst_exponent,
        [Arg(DF), Arg(INT), Arg(INT, default=2), Arg(INT, default=20)],
        DF,
    ),
}


def _build() -> tuple[dict[str, Callable[..., Any]], dict[str, dict[str, Any]]]:
    """Derive the runtime and schema dicts from `_SPECS`, validating arg ordering.

    Returns:
        A `(functions, function_schema)` pair sharing identical keys.

    Raises:
        ValueError: If any operator declares a required argument after an optional one.
    """
    functions: dict[str, Callable[..., Any]] = {}
    function_schema: dict[str, dict[str, Any]] = {}
    for name, (func, args, ret) in _SPECS.items():
        seen_optional = False
        for arg in args:
            if not arg.required:
                seen_optional = True
            elif seen_optional:
                raise ValueError(f"'{name}': required argument after optional argument")
        functions[name] = func
        function_schema[name] = {"args": args, "return": ret}
    return functions, function_schema


BUILTIN_FUNCTIONS, BUILTIN_FUNCTION_SCHEMA = _build()
