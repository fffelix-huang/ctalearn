from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from ctalearn.core.dataframe import DataFrame
from ctalearn.metrics._performance_metrics import (
    SECONDS_PER_YEAR,
    all_performance_metrics,
    annual_return,
    calmar_ratio,
    conditional_value_at_risk,
    fitness,
    kurtosis,
    longest_drawdown_duration,
    margin,
    max_drawdown,
    portfolio_weights_to_pnl,
    rank_ic,
    sharpe_ratio,
    simulate_trade,
    skewness,
    sortino_ratio,
    sr_mdd_ratio,
    turnover_ratio,
    value_at_risk,
    winrate,
)
from ctalearn.testing.dataframe import assert_df_equal


def create_df(data_dict: dict[str, Any], freq: str = "1d") -> DataFrame:
    length = len(next(iter(data_dict.values())))
    # Calculate seconds for frequency
    if freq.endswith("d"):
        freq_int = int(freq[:-1]) * 86400
    elif freq.endswith("h"):
        freq_int = int(freq[:-1]) * 3600
    elif freq.endswith("s"):
        freq_int = int(freq[:-1])
    else:
        freq_int = 86400  # Default to 1d

    dates = pl.datetime_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1) + timedelta(seconds=(length - 1) * freq_int),
        interval=freq,
        eager=True,
    )
    pl_df = pl.DataFrame({**data_dict, "ts": dates})
    # Explicitly cast to Float64
    pl_df = pl_df.with_columns(pl.exclude("ts").cast(pl.Float64))
    return DataFrame(pl_df, "ts", freq=freq_int)


class TestSimulateTrade:
    def test_non_dataframe_raises(self) -> None:
        prices = create_df({"asset": [100, 110]}, freq="1d")
        not_a_df: Any = [1.0, 1.0]
        with pytest.raises(TypeError):
            simulate_trade(not_a_df, prices, fee=0.0, initial_cash=100.0)

    def test_basic(self) -> None:
        prices = create_df({"asset": [100, 110, 121]}, freq="1d")
        weights = create_df({"asset": [1.0, 1.0, 1.0]}, freq="1d")
        result = simulate_trade(weights, prices, fee=0.01, initial_cash=100.0)
        expected = create_df(
            {"nav": [99.0, 108.9, 119.79], "turnover": [100.0, 0.0, 0.0]}
        )
        assert_df_equal(result, expected)

    def test_not_aligned(self) -> None:
        prices = create_df({"asset": [100, 110, 121]}, freq="2d")
        weights = create_df({"asset": [0.1, 0.2, 0.3, 0.4, 0.5]}, freq="1d")
        result = simulate_trade(weights, prices, fee=0.0, initial_cash=100.0)
        expected = create_df(
            {"nav": [100.0, 101.0, 104.03], "turnover": [10.0, 19.3, 18.685]}, freq="2d"
        )
        assert_df_equal(result, expected)

    def test_different_length(self) -> None:
        prices = create_df({"asset": [100, 110, 121, 110, 100]}, freq="2d")
        weights = create_df({"asset": [0.1, 0.2, 0.3, 0.4, 0.5]}, freq="1d")
        result = simulate_trade(weights, prices, fee=0.0, initial_cash=100.0)
        expected = create_df(
            {"nav": [100.0, 101.0, 104.03], "turnover": [10.0, 19.3, 18.685]}, freq="2d"
        )
        assert_df_equal(result, expected)

    def test_weight_rebalance(self) -> None:
        prices = create_df({"asset": [100, 120, 108]}, freq="1d")
        weights = create_df({"asset": [0.5, 0.0]}, freq="2d")
        result = simulate_trade(weights, prices, fee=0.0, initial_cash=100.0)
        expected = create_df(
            {"nav": [100.0, 110.0, 104.5], "turnover": [50.0, 5.0, 49.5]}
        )
        assert_df_equal(result, expected)

    def test_weight_rebalance_with_fee(self) -> None:
        prices = create_df({"asset": [100, 120, 108]}, freq="1d")
        weights = create_df({"asset": [0.5, 0.0]}, freq="2d")
        result = simulate_trade(weights, prices, fee=0.01, initial_cash=100.0)
        expected = create_df(
            {"nav": [99.5, 109.353, 103.3907], "turnover": [50.0, 4.7, 49.23]},
            freq="1d",
        )
        assert_df_equal(result, expected)

    def test_multi_asset(self) -> None:
        prices = create_df({"a": [100, 110, 121], "b": [100, 90, 81]}, freq="1d")
        weights = create_df({"a": [0.5, 0.5, 0.5], "b": [0.5, 0.5, 0.5]}, freq="1d")
        result = simulate_trade(weights, prices, fee=0.0, initial_cash=100.0)
        expected = create_df(
            {"nav": [100.0, 100.0, 100.0], "turnover": [100.0, 10.0, 10.0]}, freq="1d"
        )
        assert_df_equal(result, expected)

    def test_complex(self) -> None:
        prices = create_df(
            {"a": [100, 102, 101, 104, 99], "b": [30, 29, 28, 31, 30]}, freq="1d"
        )
        weights = create_df(
            {"a": [0.2, 0.25, 0.3, 0.3, 0.3], "b": [-0.1, -0.1, 0.2, 0.2, 0.0]},
            freq="1d",
        )
        result = simulate_trade(weights, prices, fee=0.0005, initial_cash=100.0)
        expected = create_df(
            {
                "nav": [99.985, 100.71553589, 100.79840401, 103.8546513, 101.67622768],
                "turnover": [30.0, 5.19488, 35.2022082, 1.55211594, 20.94835534],
            },
            freq="1d",
        )
        assert_df_equal(result, expected)

    def test_empty_weights_raises(self) -> None:
        prices = create_df({"asset": [100, 110]}, freq="1d")
        weights = DataFrame(
            pl.DataFrame(schema={"ts": pl.Datetime, "asset": pl.Float64}),
            "ts",
            freq=86400,
            _skip_validate=True,
        )
        with pytest.raises(ValueError, match="Weights DataFrame is empty"):
            simulate_trade(weights, prices, fee=0.0, initial_cash=100.0)

    def test_no_common_columns_raises(self) -> None:
        prices = create_df({"price_only": [100, 110]}, freq="1d")
        weights = create_df({"weight_only": [1.0, 1.0]}, freq="1d")
        with pytest.raises(ValueError, match="No common asset columns"):
            simulate_trade(weights, prices, fee=0.0, initial_cash=100.0)

    @pytest.mark.benchmark(group="simulate_trade", min_rounds=100, disable_gc=True)
    def test_benchmark_500000(self, benchmark: Any) -> None:
        np.random.seed(42)
        prices = create_df(
            {
                "a": np.random.uniform(low=1, high=1000, size=500000),
                "b": np.random.uniform(low=1, high=10000, size=500000),
            }
        )
        weights = create_df(
            {
                "a": np.random.uniform(low=0, high=0.5, size=500000),
                "b": np.random.uniform(low=0, high=0.5, size=500000),
            }
        )
        _ = benchmark(
            simulate_trade,
            weights,
            prices,
            fee=0.0005,
            initial_cash=1000.0,
        )


class TestPortfolioWeightsToPnl:
    def test_basic_hold(self) -> None:
        # Scenario: Buy and Hold
        # Prices: 100 -> 110 (10%) -> 121 (10%)
        prices = create_df({"asset": [100, 110, 121]}, freq="1d")
        weights = create_df({"asset": [1.0, 1.0, 1.0]}, freq="1d")

        pnl = portfolio_weights_to_pnl(weights, prices, fee=0.0)
        expected = create_df({"pnl": [0.0, 0.1, 0.1]})

        assert_df_equal(pnl, expected)

    def test_with_fee_and_trading(self) -> None:
        prices = create_df({"asset": [100, 110, 110]}, freq="1d")
        weights = create_df({"asset": [0.5, 1.0, 1.0]}, freq="1d")
        fee = 0.01

        pnl = portfolio_weights_to_pnl(weights, prices, fee=fee)
        expected = create_df({"pnl": [-0.005, 0.04472361809, 0.0]})

        assert_df_equal(pnl, expected)

    def test_mixed_frequency(self) -> None:
        # Weights (2d) vs Prices (1d)
        prices = create_df({"a": [100.0, 101.0, 102.0, 103.0, 104.0]}, freq="1d")
        weights = create_df({"a": [1.0, 0.0, 1.0]}, freq="2d")

        pnl = portfolio_weights_to_pnl(weights, prices, fee=0.0)
        expected = create_df({"pnl": [0.0, 101 / 100 - 1, 102 / 101 - 1, 0.0, 0.0]})

        assert_df_equal(pnl, expected)

    def test_non_dataframe_raises(self) -> None:
        prices = create_df({"asset": [100, 110]}, freq="1d")
        not_a_df: Any = [1.0, 1.0]
        with pytest.raises(TypeError, match="must be ctalearn DataFrame"):
            portfolio_weights_to_pnl(not_a_df, prices)


class TestSharpeRatio:
    def test_constant_pnl(self) -> None:
        # Std is 0 -> Sharpe 0
        pnl = create_df({"pnl": [0.01] * 5}, freq="1d")
        np.testing.assert_almost_equal(sharpe_ratio(pnl), 0.0)

    def test_volatile_pnl(self) -> None:
        # Freq 1d -> Factor 365
        pnl = create_df({"pnl": [-0.01, 0.03]}, freq="1d")
        sr = sharpe_ratio(pnl)

        mean = 0.01
        std = np.std([-0.01, 0.03], ddof=1)
        expected = (mean / std) * np.sqrt(365.0)
        np.testing.assert_almost_equal(sr, expected)


class TestSortinoRatio:
    def test_basic(self) -> None:
        pnl = create_df({"pnl": [0.01, 0.02, -0.015, -0.008, 0.007]}, freq="1d")
        result = sortino_ratio(pnl)
        expected = 4.45010788
        np.testing.assert_almost_equal(result, expected)

    def test_no_drawdown(self) -> None:
        pnl = create_df({"pnl": [0.01] * 5})
        result = sortino_ratio(pnl)
        np.testing.assert_almost_equal(result, 0.0)


class TestMaxDrawdown:
    def test_basic_drawdown(self) -> None:
        # PnL: 0, 1, -0.5, 1.5, -1
        # Cum: 0, 1, 0.5, 2.0, 1.0
        # Peak:0, 1, 1.0, 2.0, 2.0
        # DD:  0, 0, 0.5, 0.0, 1.0
        # MaxDD: 1.0
        pnl = create_df({"pnl": [0.0, 1.0, -0.5, 1.5, -1]}, freq="1d")
        np.testing.assert_almost_equal(max_drawdown(pnl), 1.0)

    def test_no_drawdown(self) -> None:
        pnl = create_df({"pnl": [0.1, 0.1, 0.1]}, freq="1d")
        np.testing.assert_almost_equal(max_drawdown(pnl), 0.0)


class TestAnnualReturn:
    def test_annual_return(self) -> None:
        pnl = create_df({"pnl": [0.01, 0.01]}, freq="1d")
        cagr = annual_return(pnl)
        expected = 1.01**365 - 1
        np.testing.assert_almost_equal(cagr, expected)


class TestCalmarRatio:
    def test_basic_calmar(self) -> None:
        pnl = create_df({"pnl": [0.02, -0.01, 0.02]}, freq="1d")

        calmar = calmar_ratio(pnl)

        returns = np.array([0.02, -0.01, 0.02])
        ann_ret = np.exp(np.mean(np.log(returns + 1)) * 365.0) - 1
        mdd = 0.01
        np.testing.assert_almost_equal(calmar, ann_ret / mdd)

    def test_zero_mdd(self) -> None:
        pnl = create_df({"pnl": [0.1, 0.1]}, freq="1d")
        np.testing.assert_almost_equal(calmar_ratio(pnl), 0.0)


class TestTurnoverRatio:
    def test_static_turnover(self) -> None:
        prices = create_df({"a": [100, 100, 100]}, freq="1d")
        weights = create_df({"a": [0.5, -0.5, 0.0]}, freq="1d")

        result = turnover_ratio(weights, prices)
        np.testing.assert_almost_equal(result, (0.5 + 1.0 + 0.5) / 3)

    def test_basic_turnover(self) -> None:
        prices = create_df({"a": [100, 150, 100]}, freq="1d")
        weights = create_df({"a": [0.5, 0.0, 0.5]}, freq="1d")

        result = turnover_ratio(weights, prices)
        np.testing.assert_almost_equal(result, (0.5 + 0.6 + 0.5) / 3)

    def test_multi_asset_turnover(self) -> None:
        prices = create_df({"a": [100, 100, 100], "b": [100, 150, 100]}, freq="1d")
        weights = create_df({"a": [0.5, -0.5, 0.0], "b": [0.5, 0.0, 0.5]}, freq="1d")

        day_1 = (0.5 + 0.5) / 1
        day_2 = (1.125 + 0.75) / 1.25
        day_3 = (0.625 + 0.625) / 1.25

        result = turnover_ratio(weights, prices)
        np.testing.assert_almost_equal(result, (day_1 + day_2 + day_3) / 3)


class TestMargin:
    def test_basic(self) -> None:
        prices = create_df({"asset": [100, 110, 121]}, freq="1d")
        weights = create_df({"asset": [1.0, 1.0, 1.0]}, freq="1d")
        result = simulate_trade(weights, prices, fee=0.01, initial_cash=100.0)
        expected = create_df(
            {"nav": [99.0, 108.9, 119.79], "turnover": [100.0, 0.0, 0.0]}
        )
        assert_df_equal(result, expected)

        m = margin(weights, prices, fee=0.01)
        np.testing.assert_almost_equal(m, 0.099)

    def test_no_trades(self) -> None:
        # Zero weights -> never trade -> turnover 0, nav flat -> margin = 0/0 = NaN
        # every row. The filter drops all rows, so mean() is None. Should be 0.0,
        # not a TypeError from float(None).
        prices = create_df({"asset": [100, 110, 121]}, freq="1d")
        weights = create_df({"asset": [0.0, 0.0, 0.0]}, freq="1d")

        m = margin(weights, prices, fee=0.01)
        np.testing.assert_almost_equal(m, 0.0)


class TestSrMddRatio:
    def test_ratio(self) -> None:
        # Just verifies division logic
        pnl = create_df({"pnl": [0.02, -0.01]}, freq="1d")
        sr = sharpe_ratio(pnl)
        mdd = max_drawdown(pnl)
        expected = sr / mdd

        result = sr_mdd_ratio(pnl)
        np.testing.assert_almost_equal(result, expected)

    def test_zero_mdd(self) -> None:
        # No drawdown -> mdd 0 -> guard returns 0.0 instead of dividing.
        pnl = create_df({"pnl": [0.1, 0.1, 0.1]}, freq="1d")
        np.testing.assert_almost_equal(sr_mdd_ratio(pnl), 0.0)


class TestLongestDrawdownDuration:
    def test_ldd_days(self) -> None:
        # Freq = 1d.
        # PnL sequence creating drawdown of length 2 steps.
        # Cum: 0 -> 1 -> 0.9 -> 0.8 -> 1.1
        # Peaks: 0, 1, 1, 1, 1.1
        # InDD: F, F, T, T, F
        # Max Run: 2 steps.
        # Result should be 2 days.
        pnl = create_df({"pnl": [0.0, 1.0, -0.1, -0.1, 0.3]}, freq="1d")

        ldd = longest_drawdown_duration(pnl)
        np.testing.assert_almost_equal(ldd, 2.0)

    def test_ldd_seconds(self) -> None:
        # Freq = 1s.
        # Factor = 31536000.
        # Run = 2 steps.
        # Days = (2 / 31536000) * 365.
        # Which is simply: 2 seconds converted to days -> 2 / 86400 days.
        pnl = create_df({"pnl": [0.0, 1.0, -0.1, -0.1, 0.3]}, freq="1s")

        ldd = longest_drawdown_duration(pnl)
        expected_days = 2 / 86400
        np.testing.assert_almost_equal(ldd, expected_days)

    def test_ldd_zero_factor(self) -> None:
        # factor == 0 (freq unknown) must not divide-by-zero; return 0.0.
        pnl = create_df({"pnl": [0.0, 1.0, -0.1, -0.1, 0.3]}, freq="1d")
        pnl.freq = 0
        assert longest_drawdown_duration(pnl) == 0.0

    def test_no_drawdown(self) -> None:
        # Monotonic equity -> never in drawdown -> max duration None -> 0.0.
        pnl = create_df({"pnl": [0.1, 0.1, 0.1]}, freq="1d")
        assert longest_drawdown_duration(pnl) == 0.0


class TestWinrate:
    def test_winrate(self) -> None:
        # >0: 2. !=0: 4. WR=0.5
        pnl = create_df({"pnl": [0.1, -0.1, 0.0, 0.2, -0.2]}, freq="1d")
        np.testing.assert_almost_equal(winrate(pnl), 0.5)

    def test_no_trades(self) -> None:
        pnl = create_df({"pnl": [0.0, 0.0]}, freq="1d")
        np.testing.assert_almost_equal(winrate(pnl), 0.0)


class TestRankIC:
    def spearman_corr(self, x_ranks: list[float], y_ranks: list[float]) -> float:
        n = len(x_ranks)
        avg_x = sum(x_ranks) / n
        avg_y = sum(y_ranks) / n

        numerator = sum((xi - avg_x) * (yi - avg_y) for xi, yi in zip(x_ranks, y_ranks))
        denom_x = sum((xi - avg_x) ** 2 for xi in x_ranks)
        denom_y = sum((yi - avg_y) ** 2 for yi in y_ranks)

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return float(numerator / np.sqrt(denom_x * denom_y))

    def test_basic(self) -> None:
        signals = create_df(
            {
                "ADA": [1.0, 1.0, -1.0],
                "BNB": [2.0, 0.5, 0.3],
                "BTC": [0.5, 0.2, -0.5],
                "ETH": [-0.5, 0.1, 0.5],
                "SOL": [0.2, -0.1, 0.3],
            },
            freq="1d",
        )
        prices = create_df(
            {
                "ADA": [50, 52, 53, 49],
                "BNB": [100, 99, 98, 97],
                "BTC": [100, 102, 101, 99],
                "ETH": [49, 51, 52, 53],
                "SOL": [10, 11, 9, 12],
            },
            freq="1d",
        )

        result = rank_ic(signals, prices, horizon=1)
        expected = create_df(
            {
                "rank_ic": [
                    self.spearman_corr([4, 5, 3, 1, 2], [3, 1, 2, 4, 5]),
                    self.spearman_corr([5, 4, 3, 2, 1], [4, 2, 3, 5, 1]),
                    self.spearman_corr([1, 3.5, 2, 5, 3.5], [1, 3, 2, 4, 5]),
                ]
            }
        )

        assert_df_equal(result, expected)

    def test_not_aligned(self) -> None:
        signals = create_df(
            {
                "ADA": [1.0, 1.0, -1.0],
                "BNB": [2.0, 0.5, 0.3],
                "BTC": [0.5, 0.2, -0.5],
                "ETH": [-0.5, 0.1, 0.5],
                "SOL": [0.2, -0.1, 0.3],
            },
            freq="1d",
        ).shift_time(-timedelta(days=1))
        prices = create_df(
            {
                "ADA": [50, 52, 53, 49],
                "BNB": [100, 99, 98, 97],
                "BTC": [100, 102, 101, 99],
                "ETH": [49, 51, 52, 53],
                "SOL": [10, 11, 9, 12],
            },
            freq="1d",
        )

        result = rank_ic(signals, prices, horizon=1)
        expected = create_df(
            {
                "rank_ic": [
                    self.spearman_corr([5, 4, 3, 2, 1], [3, 1, 2, 4, 5]),
                    self.spearman_corr([1, 3.5, 2, 5, 3.5], [4, 2, 3, 5, 1]),
                ]
            }
        )

        assert_df_equal(result, expected)

    def test_different_frequency(self) -> None:
        signals = create_df(
            {
                "ADA": [1.0, 1.0, -1.0],
                "BNB": [2.0, 0.5, 0.3],
                "BTC": [0.5, 0.2, -0.5],
                "ETH": [-0.5, 0.1, 0.5],
                "SOL": [0.2, -0.1, 0.3],
            },
            freq="12h",
        )
        prices = create_df(
            {
                "ADA": [50, 52, 53, 49],
                "BNB": [100, 99, 98, 97],
                "BTC": [100, 102, 101, 99],
                "ETH": [49, 51, 52, 53],
                "SOL": [10, 11, 9, 12],
            },
            freq="1d",
        )

        result = rank_ic(signals, prices, horizon=1)
        expected = create_df(
            {
                "rank_ic": [
                    self.spearman_corr([4, 5, 3, 1, 2], [3, 1, 2, 4, 5]),
                    self.spearman_corr([1, 3.5, 2, 5, 3.5], [4, 2, 3, 5, 1]),
                ]
            }
        )

        assert_df_equal(result, expected)

    def test_horizon(self) -> None:
        signals = create_df(
            {
                "ADA": [1.0, 1.0, -1.0],
                "BNB": [2.0, 0.5, 0.3],
                "BTC": [0.5, 0.2, -0.5],
                "ETH": [-0.5, 0.1, 0.5],
                "SOL": [0.2, -0.1, 0.3],
            },
            freq="1d",
        )
        prices = create_df(
            {
                "ADA": [50, 52, 53, 49],
                "BNB": [100, 99, 98, 97],
                "BTC": [100, 102, 101, 99],
                "ETH": [49, 51, 52, 53],
                "SOL": [10, 11, 9, 12],
            },
            freq="1d",
        )

        result = rank_ic(signals, prices, horizon=2)
        expected = create_df(
            {
                "rank_ic": [
                    self.spearman_corr([4, 5, 3, 1, 2], [5, 1, 3, 4, 2]),
                    self.spearman_corr([5, 4, 3, 2, 1], [1, 3, 2, 4, 5]),
                ]
            }
        )

        assert_df_equal(result, expected)

    def test_non_dataframe_raises(self) -> None:
        prices = create_df({"AAA": [100.0, 110.0]}, freq="1d")
        not_a_df: Any = [1.0, 2.0]
        with pytest.raises(TypeError, match="must be ctalearn DataFrame"):
            rank_ic(not_a_df, prices)

    def test_horizon_not_int_raises(self) -> None:
        df = create_df({"AAA": [1.0, 2.0]}, freq="1d")
        with pytest.raises(TypeError, match="'horizon' should be int"):
            rank_ic(df, df, horizon=1.5)  # type: ignore[arg-type]

    def test_horizon_non_positive_raises(self) -> None:
        df = create_df({"AAA": [1.0, 2.0]}, freq="1d")
        with pytest.raises(ValueError, match="'horizon' should be greater than 0"):
            rank_ic(df, df, horizon=0)

    def test_no_common_assets(self) -> None:
        signals = create_df({"AAA": [1.0, 2.0, 3.0]}, freq="1d")
        prices = create_df({"BBB": [100.0, 110.0, 121.0]}, freq="1d")
        result = rank_ic(signals, prices, horizon=1)
        assert len(result._df) == 0

    def test_empty_future_returns(self) -> None:
        # Signal window sits entirely outside the price-return window -> empty.
        prices = create_df({"AAA": [100.0, 110.0, 121.0]}, freq="1d")
        signals = create_df({"AAA": [1.0, 2.0]}, freq="1d").shift_time(
            timedelta(days=365)
        )
        result = rank_ic(signals, prices, horizon=1)
        assert len(result._df) == 0


class TestValueAtRisk:
    def test_basic(self) -> None:
        log_returns = create_df({"pnl": [0.04, -0.05, 0.02, -0.01]})
        var95 = value_at_risk(log_returns, alpha=0.95)
        np.testing.assert_almost_equal(var95, -0.044)

    def test_no_drawdown(self) -> None:
        log_returns = create_df({"pnl": [0.04, 0.05, 0.02, 0.01]})
        var95 = value_at_risk(log_returns, alpha=0.95)
        np.testing.assert_almost_equal(var95, 0.0115)


class TestConditionalValueAtRisk:
    def test_basic(self) -> None:
        log_returns = create_df({"pnl": [0.04, -0.05, 0.02, -0.01]})
        cvar95 = conditional_value_at_risk(log_returns, alpha=0.95)
        np.testing.assert_almost_equal(cvar95, -0.05)

    def test_no_drawdown(self) -> None:
        log_returns = create_df({"pnl": [0.04, 0.05, 0.02, 0.01]})
        cvar95 = conditional_value_at_risk(log_returns, alpha=0.95)
        np.testing.assert_almost_equal(cvar95, 0.01)


class TestSkewness:
    def test_basic(self) -> None:
        pnl = create_df({"pnl": [0.01, -0.02, 0.03, -0.01, 0.05]}, freq="1d")
        expected = pnl._df.select(pl.col("pnl").skew()).item()
        np.testing.assert_almost_equal(skewness(pnl), expected)


class TestKurtosis:
    def test_basic(self) -> None:
        pnl = create_df({"pnl": [0.01, -0.02, 0.03, -0.01, 0.05]}, freq="1d")
        expected = pnl._df.select(pl.col("pnl").kurtosis(fisher=True)).item()
        np.testing.assert_almost_equal(kurtosis(pnl, fisher=True), expected)


class TestFitness:
    def test_basic(self) -> None:
        prices = create_df({"asset": [100, 110, 121, 115]}, freq="1d")
        weights = create_df({"asset": [1.0, 1.0, 1.0, 1.0]}, freq="1d")
        result = fitness(weights, prices, fee=0.0)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestAllPerformanceMetrics:
    def test_returns_all_metrics(self) -> None:
        prices = create_df({"asset": [100, 110, 121, 115]}, freq="1d")
        weights = create_df({"asset": [1.0, 1.0, 1.0, 1.0]}, freq="1d")
        result = all_performance_metrics(weights, prices, fee=0.0)

        assert set(result) == {
            "sharpe",
            "sortino",
            "mdd",
            "annual_return",
            "calmar",
            "turnover_ratio",
            "sr_mdd_ratio",
            "longest_drawdown_duration",
            "winrate",
            "fitness",
            "margin",
            "skewness",
            "kurtosis",
        }
        assert all(isinstance(v, float) for v in result.values())
