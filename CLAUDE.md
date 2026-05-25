# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # install / sync dependencies
uv run pytest                    # run tests (doctests + coverage; benchmarks skipped)
uv run pytest --benchmark-only   # run only the benchmarks
uv run pytest tests/operator/test_cross_sectional.py              # single file
uv run pytest tests/operator/test_cross_sectional.py::test_cs_rank  # single test
uv run ruff check --fix          # lint (isort only — ruff select = ["I"])
uv run ruff format               # format
uv run pre-commit run --all-files  # run all pre-commit hooks
uv run --group docs mkdocs serve   # live docs preview at localhost:8000
uv run --only-group docs mkdocs build --strict  # build docs (CI uses this; --strict fails on broken refs)
```

`pytest` always runs with `--doctest-modules --cov=ctalearn --cov-report=term-missing --benchmark-skip` (see `pytest.ini`), so coverage (scoped to the package) and in-source doctests run on every invocation, while `pytest-benchmark` tests are skipped by default — pass `--benchmark-only` to run them (the only flag that overrides `--benchmark-skip`; `--benchmark-enable` does not).

## Architecture

The library is a cross-sectional / time-series alpha research toolkit. Everything is built around a custom `DataFrame` class that wraps a Polars `pl.DataFrame`.

### `ctalearn/core/dataframe.py` — the central abstraction

`DataFrame(df, time_col, freq, name)` stores:
- `_df`: the underlying `pl.DataFrame`
- `time_col`: name of the timestamp column
- `freq`: integer seconds between bars (auto-detected from min diff, or supplied)
- `name`: lineage string that accumulates operator call history (for debugging)

On construction the class sorts by time, validates regular sampling, and calls `_ensure_dense` to fill gaps using `join_asof`. This dense-grid guarantee is what makes the fast-path in `@auto_align` safe: if two DataFrames share the same `freq`, start, and end they are assumed to be row-aligned and no explicit join is performed.

Arithmetic operators (`+`, `-`, `*`, `/`) use the `@auto_align` decorator which inner-joins on the time column before delegating to the concrete `_*_aligned` method. Scalar operands bypass alignment.

`__getattr__` proxies unknown attribute access to `_df`, wrapping any returned `pl.DataFrame` back into a `DataFrame`. This lets callers call most Polars methods transparently. Internal construction that is already validated uses `_skip_validate=True` to avoid redundant O(N) work.

A `DataFrame` holds one of two shapes: a **wide panel** (time + N asset columns: prices, signals, weights) or a **single series** (time + one value column: pnl, nav). Series-shaped consumers must locate their value column via the `value_col` property — never a hardcoded name like `"pnl"`. `value_col` returns the lone non-time column and raises if the frame isn't single-series.

### `ctalearn/operator/` — mathematical operators

Four modules, all accepting and returning `DataFrame`:

| Module | Prefix | Purpose |
|---|---|---|
| `_arithmetic.py` | — | Element-wise transforms: `log`, `sqrt`, `cbrt`, `sign`, `symmetric_log`, etc. |
| `_cross_sectional.py` | `cs_` | Row-wise (across assets): `cs_rank`, `cs_zscore`, `cs_winsorize`, `cs_mean`, `cs_pca`, `vector_neut`, `regression_neut`, `process_alpha_weights` |
| `_time_series.py` | `ts_` | Column-wise (through time): `ts_rank`, `ts_mean`, `ts_std_dev`, `ts_delta`, `ts_corr`, `ts_zscore`, `ts_hurst_exponent`, etc. |
| `_strategy.py` | — | Signal-to-weight strategies: `trend`, `trend_reverse`, `trend_mean_reversion`, etc. No enforced prefix — current entry-exit strategies happen to start with `trend_`, but that's not a guaranteed convention. |

Cross-sectional operators convert to NumPy for row-wise work. Time-series operators prefer native Polars (rolling) APIs; Numba (`@njit`) is a fallback used only where Polars has no equivalent (e.g. custom rolling rank/std as ring-buffer loops over `map_batches`).

### `ctalearn/dsl/` — alpha expression DSL

A Lark LALR grammar (`grammar.lark`) that parses expressions like:
```
x = ts_mean(close, 20);
return cs_rank(x - ts_mean(open, 5));
```

Pipeline: `parser` → parse tree → `TypeCheckTransformer` (static type analysis using `DslType` enum) → `ExecutionTransformer` (runtime execution).

`ExecutionTransformer` takes:
- `functions`: dict of callable operators (e.g. `{"ts_mean": ts_mean, ...}`)
- `data_loaders`: dict of lazy callables that fetch data on first access (e.g. `{"close": lambda: fetch_close()}`)

### `ctalearn/data/_fetch.py` — data ingestion

`fetch_glassnode(endpoint, params, api_key, cache_dir)` hits the Glassnode REST API and returns a `DataFrame` with a `timestamp` column. Results are cached as Parquet files in a sharded directory under `cache_dir` (e.g. `.glassnode_cache/`). `GLASSNODE_API_KEY` env var is used when `api_key` is not passed.

`fetch_glassnode_cs(endpoint, params, universe, ...)` fetches multiple assets in parallel via `ThreadPoolExecutor` and returns `dict[metric_name, DataFrame]` where each DataFrame has one column per asset.

### `ctalearn/metrics/_performance_metrics.py`

`simulate_trade(weights, prices, fee, initial_cash)` runs a Numba-accelerated backtest. `weights` and `prices` are `DataFrame` instances; the function aligns them and computes portfolio P&L.

### `ctalearn/preprocess/_split.py`

`train_test_split(df, train_size=0.7)` does a sequential (non-shuffled) split — appropriate for time series.

### `ctalearn/testing/dataframe.py`

`assert_df_equal(df1, df2)` sorts columns before comparison so column order is irrelevant.

## Docs

MkDocs + Material + mkdocstrings (numpy docstring style). Config in `mkdocs.yml`; pages in `docs/`. `docs/api/*.md` are thin `:::` directives that render each module's public API from its docstrings — no hand-written API prose. `.github/workflows/docs.yml` deploys to GitHub Pages on push to master. Docs deps live in the `docs` dependency group (not `dev`/`test`).

`docs/dsl.md` embeds `examples/dsl_interpreter.py` via a `--8<--` snippet (stays live), but its builtins **signature table is a manual paste** — regenerate from `BUILTIN_FUNCTION_SCHEMA` if `dsl/library.py::_SPECS` changes.
