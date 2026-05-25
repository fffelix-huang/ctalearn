# ctalearn

[![CI](https://github.com/fffelix-huang/ctalearn/actions/workflows/ci.yml/badge.svg)](https://github.com/fffelix-huang/ctalearn/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ctalearn.svg)](https://pypi.org/project/ctalearn/)
[![Python](https://img.shields.io/pypi/pyversions/ctalearn.svg)](https://pypi.org/project/ctalearn/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A cross-sectional / time-series **alpha research toolkit** for CTA strategies.

**ctalearn** is built around a thin `DataFrame` wrapper over [Polars](https://pola.rs)
that guarantees a dense, regularly-sampled time grid. On top of it sits a library
of vectorized operators (cross-sectional, time-series, arithmetic, strategy),
a Numba-accelerated backtester, Glassnode data ingestion, and a small embedded
DSL for expressing alphas as plain text.

## Key features

- **Aligned `DataFrame`** — wraps `pl.DataFrame`, auto-detects bar frequency,
  fills gaps to a dense grid, and aligns operands automatically on arithmetic
  (`+ - * /`).
- **Operator library** — `cs_*` (across assets), `ts_*` (through time),
  element-wise transforms, and signal-to-weight strategies. Native
  Polars where possible; Numba only where Polars has no equivalent.
- **Alpha DSL** — a Lark grammar for expressions like
  `cs_rank(ts_mean(close, 20) - open)`, with static type checking before
  execution. Ships its built-in function registry so consumers only provide
  their own data loaders.
- **Backtesting** — Numba-accelerated `simulate_trade(weights, prices, fee)`.
- **Data** — cached Glassnode REST ingestion, single-asset and parallel
  cross-sectional fetch.

## Installation

Requires Python 3.11+.

```bash
pip install ctalearn
uv add ctalearn        # with uv
```

## Quick start

```python
import polars as pl
from ctalearn import DataFrame
from ctalearn.operator import ts_mean, cs_rank

prices = DataFrame(
    pl.DataFrame({
        "ts":  pl.datetime_range(..., interval="1d", eager=True),
        "BTC": [...],
        "ETH": [...],
    }),
    time_col="ts",
)

# 20-bar moving average, then rank across assets each bar.
signal = cs_rank(prices - ts_mean(prices, 20))
```

### Alpha DSL

Express the same idea as text and run it through the interpreter:

```python
from ctalearn.dsl import (
    parser, TypeCheckTransformer, ExecutionTransformer,
    BUILTIN_FUNCTIONS, BUILTIN_FUNCTION_SCHEMA, DslType,
)

code = "return cs_rank(close - ts_mean(close, 20));"

# Caller supplies only the factor side; functions come from the library.
factor_schema = {"close": DslType.DATAFRAME}
data_loaders  = {"close": lambda: prices}

tree   = parser.parse(code)
TypeCheckTransformer(factor_schema, BUILTIN_FUNCTION_SCHEMA).transform(tree)
result = ExecutionTransformer(BUILTIN_FUNCTIONS, data_loaders).transform(tree)
```

See [`examples/dsl_interpreter.py`](examples/dsl_interpreter.py) for a complete,
runnable integration.

## Development

```bash
git clone https://github.com/fffelix-huang/ctalearn.git
cd ctalearn
uv sync                            # install / sync dependencies
uv run pytest                      # run tests (doctests + coverage; benchmarks skipped)
uv run pytest --benchmark-only     # run only the benchmarks
uv run ruff check --fix            # lint
uv run ruff format                 # format
uv run pre-commit run --all-files  # all hooks
```

## License

[MIT](LICENSE)
