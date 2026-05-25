# ctalearn

A cross-sectional / time-series alpha research toolkit for CTA strategies, built
around a Polars-backed [`DataFrame`](api/dataframe.md) abstraction.

## Install

```bash
pip install ctalearn
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add ctalearn        # into a project
uv pip install ctalearn  # into the active environment
```

## Quick start

```python
import polars as pl
from ctalearn import DataFrame
from ctalearn.operator import cs_rank, ts_mean

df = DataFrame(pl.read_csv("prices.csv"), time_col="timestamp")
signal = cs_rank(ts_mean(df, 20))
```

See the [API Reference](api/dataframe.md) for the full operator catalog, the
[DSL Guide](dsl.md) for writing alpha expressions as text, and [Data](api/data.md)
for Glassnode ingestion.
