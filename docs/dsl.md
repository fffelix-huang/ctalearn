# DSL Guide

ctalearn ships a small expression language for writing alpha factors as text.
An expression is parsed, statically type-checked, then executed against your
data — operators come from the library, only the factor data is project-supplied.

For the module API (transformers, parser, schema types) see the
[DSL API reference](api/dsl.md).

## Syntax

A program is zero or more assignments followed by exactly one `return`:

```
# This is a comment
x = ts_mean(close, 20);
y = close - x;
return cs_rank(y);
```

Rules:

- **Statements end in `;`.** Each assignment binds a `CNAME` (identifier) to an
  expression. The final `return <expr>;` is required and is the program's value.
- **Operators:** `+`, `-`, `*`, `/` with the usual precedence (`*`/`/` bind
  tighter than `+`/`-`); parenthesize to override. Unary minus (`-x`) is allowed.
- **Function calls:** `name(arg, arg, ...)`. Arguments are positional only —
  there are no keyword arguments. Optional parameters are trailing and may be
  omitted to take their default (see the [builtins table](#built-in-functions)).
- **Numbers:** integer (`20`) or decimal (`1.5`) literals only. No exponent form
  (`1e5`) — the int/float split is a simple "has a dot?" check, which also keeps
  the type checker honest about `int` vs `float` parameters.
- **No bool literals.** Operators with `bool` parameters expose them with their
  Python default; they can't be set from the DSL.
- **Comments:** `#` to end of line. Whitespace and newlines are insignificant.

### Types

Static type checking uses three virtual types ([`DslType`](api/dsl.md)):
`DataFrame`, `float`, `int`. A binary op touching a `DataFrame`
broadcasts to `DataFrame`; otherwise `float` wins over `int`. Passing a `float`
where an `int` is required (e.g. a window length) is a type error caught before
any data is touched. (There's no `bool` type — the grammar has no bool literal,
and operators' `bool` params fall back to their Python defaults.)

## Pipeline

Three stages, each operating on the parse tree:

```python
from ctalearn.dsl import (
    parser,
    TypeCheckTransformer,
    ExecutionTransformer,
    BUILTIN_FUNCTIONS,
    BUILTIN_FUNCTION_SCHEMA,
    DslType,
)

tree = parser.parse(code)                                   # 1. parse
TypeCheckTransformer(factor_schema, BUILTIN_FUNCTION_SCHEMA).transform(tree)  # 2. type-check
result = ExecutionTransformer(BUILTIN_FUNCTIONS, data_loaders).transform(tree)  # 3. execute
```

You supply two project-specific maps; the function side comes from the library:

- `factor_schema: dict[str, DslType]` — names usable as variables and their types
  (e.g. `{"close": DslType.DATAFRAME}`).
- `data_loaders: dict[str, Callable[[], DataFrame]]` — lazy, zero-arg loaders
  called only when a factor is actually referenced.

## Built-in functions

You do **not** hand-write function signatures. The library exposes its operator
registry as a single source of truth:

- **`BUILTIN_FUNCTIONS`** — `dict[str, Callable]`, the runtime operators fed to
  `ExecutionTransformer`.
- **`BUILTIN_FUNCTION_SCHEMA`** — `dict[str, {"args": list[Arg], "return": DslType}]`,
  the type signatures fed to `TypeCheckTransformer`.

Both dicts share identical keys. Introspect them at runtime:

```python
from ctalearn.dsl import BUILTIN_FUNCTION_SCHEMA

spec = BUILTIN_FUNCTION_SCHEMA["ts_mean"]
spec["args"]    # [Arg(type=DslType.DATAFRAME), Arg(type=DslType.INT)]
spec["return"]  # DslType.DATAFRAME
```

Each [`Arg`](api/dsl.md) carries a `type` and `.required` (false when it has a
default). The signatures below are derived from this schema:

| Signature | Returns |
|---|---|
| `sign(DataFrame)` | `DataFrame` |
| `log(DataFrame)` | `DataFrame` |
| `symmetric_log(DataFrame)` | `DataFrame` |
| `sqrt(DataFrame)` | `DataFrame` |
| `symmetric_sqrt(DataFrame)` | `DataFrame` |
| `cbrt(DataFrame)` | `DataFrame` |
| `identity(DataFrame)` | `DataFrame` |
| `cs_mean(DataFrame)` | `DataFrame` |
| `cs_rank(DataFrame)` | `DataFrame` |
| `cs_zscore(DataFrame)` | `DataFrame` |
| `cs_winsorize(DataFrame, float)` | `DataFrame` |
| `vector_neut(DataFrame, DataFrame)` | `DataFrame` |
| `ts_rank(DataFrame, int, float=0.0)` | `DataFrame` |
| `ts_mean(DataFrame, int)` | `DataFrame` |
| `ts_median(DataFrame, int)` | `DataFrame` |
| `ts_std_dev(DataFrame, int, int=0)` | `DataFrame` |
| `ts_zscore(DataFrame, int)` | `DataFrame` |
| `ts_robust_zscore(DataFrame, int)` | `DataFrame` |
| `ts_sum(DataFrame, int)` | `DataFrame` |
| `ts_min(DataFrame, int)` | `DataFrame` |
| `ts_max(DataFrame, int)` | `DataFrame` |
| `ts_scale(DataFrame, int, float=0.0)` | `DataFrame` |
| `ts_decay_linear(DataFrame, int)` | `DataFrame` |
| `ts_delay(DataFrame, int)` | `DataFrame` |
| `ts_delta(DataFrame, int)` | `DataFrame` |
| `ts_ffill(DataFrame, int=None)` | `DataFrame` |
| `ts_corr(DataFrame, DataFrame, int)` | `DataFrame` |
| `ts_hurst_exponent(DataFrame, int, int=2, int=20)` | `DataFrame` |

!!! note "Not every operator is exposed"
    Operators the grammar can't express are omitted from the registry — e.g.
    `cs_pca` (tuple return) and `regression_neut` (a list of factor DataFrames).
    `bool` parameters are dropped from signatures and fall back to the operator's
    Python default.

## End-to-end example

```python
--8<-- "examples/dsl_interpreter.py"
```
