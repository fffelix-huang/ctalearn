from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import ctalearn.data._fetch as fetch_mod
from ctalearn.core.dataframe import DataFrame
from ctalearn.data._fetch import fetch_glassnode, fetch_glassnode_cs

# 3 daily unix timestamps from 2023-01-01 + a value column, in Glassnode CSV form.
_CSV = "timestamp,value\n1672531200,1.0\n1672617600,2.0\n1672704000,3.0\n"


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text.encode()

    def raise_for_status(self) -> None:
        pass


class _FakeClient:
    """Context-manager stand-in for httpx.Client that returns canned CSV."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def get(self, url: str, params: dict[str, str] | None = None) -> _FakeResponse:
        return _FakeResponse(_CSV)


def _make_df(symbol: str) -> DataFrame:
    ts = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(3)]
    base = {"BTC": 100.0, "ETH": 50.0}.get(symbol, 1.0)
    return DataFrame(
        pl.DataFrame({"timestamp": ts, "price": [base, base + 1, base + 2]}),
        "timestamp",
        freq=86400,
    )


class TestFetchGlassnodeCs:
    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def stub(*, params: dict[str, str], **_: Any) -> DataFrame:
            return _make_df(params["a"])

        monkeypatch.setattr(fetch_mod, "fetch_glassnode", stub)

        out = fetch_glassnode_cs("/ep", {}, universe=["BTC", "ETH"])

        assert set(out) == {"price"}
        assert set(out["price"]._df.columns) == {"timestamp", "BTC", "ETH"}

    def test_raises_on_failed_asset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # One asset fetch blows up; the universe must not come back silently short.
        def stub(*, params: dict[str, str], **_: Any) -> DataFrame:
            if params["a"] == "ETH":
                raise ValueError("boom")
            return _make_df(params["a"])

        monkeypatch.setattr(fetch_mod, "fetch_glassnode", stub)

        with pytest.raises(RuntimeError, match="ETH"):
            fetch_glassnode_cs("/ep", {}, universe=["BTC", "ETH"])


class TestFetchGlassnode:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GLASSNODE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key is required"):
            fetch_glassnode("/v1/metrics/x", {})

    def test_restricted_param_conflict_raises(self) -> None:
        with pytest.raises(ValueError, match="should be 'csv'"):
            fetch_glassnode("/v1/metrics/x", {"f": "json"}, api_key="k")

    def test_fetch_uses_env_key_and_parses_csv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # api_key falls back to the env var; no cache_dir -> always hits the API.
        monkeypatch.setenv("GLASSNODE_API_KEY", "envkey")
        monkeypatch.setattr("ctalearn.data._fetch.httpx.Client", _FakeClient)

        df = fetch_glassnode("/v1/metrics/x", {}, cache_dir=None)

        assert df.time_col == "timestamp"
        assert df._df["value"].to_list() == [1.0, 2.0, 3.0]
        assert df._df["timestamp"][0] == datetime(2023, 1, 1)

    def test_cache_miss_then_hit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First call: API + write parquet. Second: served from cache, no network.
        monkeypatch.setattr("ctalearn.data._fetch.httpx.Client", _FakeClient)
        df1 = fetch_glassnode("/v1/metrics/x", {}, api_key="k", cache_dir=tmp_path)
        assert df1._df["value"].to_list() == [1.0, 2.0, 3.0]

        def _boom(*_: Any, **__: Any) -> None:
            raise AssertionError("network must not be hit on a cache hit")

        monkeypatch.setattr("ctalearn.data._fetch.httpx.Client", _boom)
        df2 = fetch_glassnode("/v1/metrics/x", {}, api_key="k", cache_dir=tmp_path)
        assert df2._df["value"].to_list() == [1.0, 2.0, 3.0]
