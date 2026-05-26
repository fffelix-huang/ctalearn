import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from ctalearn.core.dataframe import DataFrame


def _generate_cache_key(endpoint: str, params: dict[str, str], cache_dir: Path) -> Path:
    """
    Generate a cache filename based on endpoint and params.
    """
    # Ensure param order for same hash value.
    params_str = json.dumps(params, sort_keys=True, default=str)

    raw_key = f"{endpoint}|{params_str}"
    hash_key = hashlib.sha256(raw_key.encode()).hexdigest()

    shard_dir = hash_key[:2]
    file_hash_suffix = hash_key[2:12]

    # Example: .glassnode_cache/a1/v1_metrics_market_price_usd_close-b2c3d4e5f6.parquet
    safe_name = endpoint.strip("/").replace("/", "_")

    target_dir = cache_dir / shard_dir
    target_file = target_dir / f"{safe_name}-{file_hash_suffix}.parquet"

    target_dir.mkdir(parents=True, exist_ok=True)

    return target_file


def fetch_glassnode(
    endpoint: str,
    params: dict[str, str],
    api_key: str | None = None,
    cache_dir: Path | None = None,
) -> DataFrame:
    """
    Fetch data from Glassnode and returns a DataFrame

    Parameters
    ----------
    endpoint : str
        Example: '/v1/metrics/market/price_usd_close'
    params : dict[str, str]
        Example: {'a': 'BTC', 'i': '10m'}
    api_key : str | None, default=None
        Glassnode API key. If not provided, the function attempts to load it
        from the environment variable "GLASSNODE_API_KEY".
    cache_dir : Path | None, default=None
        Directory path to store and retrieve cached parquet files.
        If None, caching is disabled and data is always fetched from the API.

    Returns
    -------
    DataFrame
        Data fetched from Glassnode in DataFrame format.
    """
    if not api_key:
        api_key = os.getenv("GLASSNODE_API_KEY", None)

        if not api_key:
            raise ValueError("api_key is required and cannot be None or empty")

    params = params.copy()

    restricted_params = {"f": "csv", "timestamp_format": "unix"}
    for key, value in restricted_params.items():
        if key not in params:
            params[key] = value
        elif params[key] != value:
            raise ValueError(
                f"Query parameter '{key}' should be '{value}' in our implementation."
            )

    name = f"fetch_glassnode('{endpoint}', {params})"

    cache_path = None

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _generate_cache_key(endpoint, params, cache_dir)

        if cache_path and cache_path.exists():
            pl_df = pl.read_parquet(cache_path)
            return DataFrame(pl_df, "timestamp", name=name)

    base_url = "https://api.glassnode.com"
    url = f"{base_url}{endpoint}"
    all_params = {**params, "api_key": api_key}

    with httpx.Client(timeout=30) as client:
        response = client.get(url, params=all_params)
        response.raise_for_status()

        # infer_schema_length=None scans the whole file before picking dtypes.
        # Glassnode columns like volume mix integer-looking and decimal values;
        # the default 100-row inference can guess i64 and then fail on a later
        # float.
        pl_df = pl.read_csv(
            io.BytesIO(response.content), infer_schema_length=None
        ).with_columns(pl.from_epoch("timestamp", time_unit="s"))

        if cache_path:
            pl_df.write_parquet(cache_path)

        return DataFrame(pl_df, "timestamp", name=name)


def fetch_glassnode_cs(
    endpoint: str,
    params: dict[str, str],
    universe: list[str],
    api_key: str | None = None,
    cache_dir: Path | None = None,
    max_workers: int = 5,
) -> dict[str, DataFrame]:
    """
    Fetch cross-sectional data from Glassnode for a universe of assets in parallel.

    This function iterates through the `universe` list, fetching data for each asset
    concurrently. It aggregates the results by metric, aligning data on the timestamp.

    Parameters
    ----------
    endpoint : str
        The Glassnode API endpoint (e.g., '/v1/metrics/market/price_usd_close').
    params : dict[str, str]
        Dictionary of query parameters. The 'a' (asset) parameter will be overridden
        by each symbol in the `universe`.
    universe : list[str]
        A list of asset symbols to fetch (e.g., ['BTC', 'ETH', 'BNB']).
    api_key : str | None, default=None
        Glassnode API key. Defaults to None (will attempt to use env var).
    cache_dir : Path | None, default=None
        Directory to use for caching requests. Defaults to None.
    max_workers : int, default=5
        Maximum number of concurrent threads for fetching data. Defaults to 5.

    Returns
    -------
    dict[str, DataFrame]
        A dictionary mapping the original metric name to a combined DataFrame.

        Example Structure:
        {
            'price': DataFrame(columns=[time_col, 'BTC', 'ETH', ...]),
            'active_addr': DataFrame(columns=[time_col, 'BTC', 'ETH', ...])
        }

        Note: The DataFrames are constructed using an 'inner' join logic across assets.
        This means the resulting DataFrame will only contain timestamps that exist
        in ALL assets.

    Raises
    ------
    RuntimeError
        If any asset in `universe` fails to fetch. The error lists every failed
        symbol and its cause; no partial result is returned, so the caller never
        gets a silently incomplete universe.

    """
    results: dict[str, DataFrame] = {}
    errors: dict[str, Exception] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol: dict[Any, str] = {}

        for symbol in universe:
            asset_params = params.copy()
            asset_params["a"] = symbol

            future = executor.submit(
                fetch_glassnode,
                endpoint=endpoint,
                params=asset_params,
                api_key=api_key,
                cache_dir=cache_dir,
            )
            future_to_symbol[future] = symbol

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]

            # Only the fetch itself may legitimately fail per-asset. Keep the
            # try this narrow so merge bugs below surface instead of being
            # mislabeled as a fetch failure.
            try:
                df = future.result()
            except Exception as e:
                errors[symbol] = e
                continue

            for col in df._df.columns:
                if col == df.time_col:
                    continue

                cur_df = df.select(
                    pl.col(df.time_col),
                    pl.col(col),
                )
                cur_df.rename({col: symbol}, inplace=True)

                existing = results.get(col)
                results[col] = (
                    existing.concat(cur_df, how="inner")
                    if existing is not None
                    else cur_df
                )

    if errors:
        detail = ", ".join(f"{s} ({e})" for s, e in errors.items())
        raise RuntimeError(f"Failed to fetch {len(errors)} asset(s): {detail}")

    return results
