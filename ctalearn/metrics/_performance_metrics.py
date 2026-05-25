from datetime import datetime, timedelta

import numpy as np
import polars as pl
from numba import njit

from ctalearn.core.dataframe import DataFrame
from ctalearn.operator._time_series import ts_delay, ts_delta, ts_ffill

SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def _get_annualize_factor(df: DataFrame) -> float:
    """Helper to calculate annualization factor from DataFrame frequency."""
    if df.freq is None or df.freq == 0:
        return 0.0
    return SECONDS_PER_YEAR / df.freq


def simulate_trade(
    weights: DataFrame, prices: DataFrame, fee: float, initial_cash: float
) -> DataFrame:
    """Simulate a continuously-rebalanced portfolio over the weight horizon.

    At every price timestamp the portfolio is rebalanced toward the most
    recent target weights (Last-Write-Wins for multiple signals before a
    price bar). Because price moves alter realized weights, rebalancing
    happens on every bar even when target weights are unchanged.

    Parameters
    ----------
    weights : DataFrame
        Target weights per asset; defines the simulation start/end.
    prices : DataFrame
        Asset prices; filtered to the weight horizon and aligned by column.
    fee : float
        Proportional trading fee per traded notional.
    initial_cash : float
        Starting cash (NAV at t0 before the first signal).

    Returns
    -------
    DataFrame
        Columns ``[time_col, "nav", "turnover"]`` on the price grid.

    Raises
    ------
    TypeError
        If `weights` or `prices` is not a DataFrame.
    ValueError
        If `weights` is empty or there are no common asset columns.
    """
    if not isinstance(weights, DataFrame) or not isinstance(prices, DataFrame):
        raise TypeError("Inputs must be ctalearn DataFrame")

    weights = weights.copy()
    prices = prices.copy()

    p_time_col = prices.time_col
    w_time_col = weights.time_col

    w_df = weights._df.sort(w_time_col)
    if w_df.is_empty():
        raise ValueError("Weights DataFrame is empty.")

    start_time = w_df[w_time_col][0]
    end_time = w_df[w_time_col][-1]

    p_df = prices._df.sort(p_time_col).filter(
        (pl.col(p_time_col) >= start_time) & (pl.col(p_time_col) <= end_time)
    )

    p_cols = set(p_df.columns)
    w_cols = set(w_df.columns)
    asset_cols = sorted(list((p_cols & w_cols) - {p_time_col, w_time_col}))

    if not asset_cols:
        raise ValueError("No common asset columns found.")

    price_times = p_df[p_time_col].cast(pl.Int64).to_numpy()
    weight_times = w_df[w_time_col].cast(pl.Int64).to_numpy()
    price_values = p_df.select(asset_cols).to_numpy().astype(np.float64)
    weight_values = w_df.select(asset_cols).to_numpy().astype(np.float64)

    @njit(  # type: ignore[untyped-decorator]
        cache=True,
        nogil=True,
        parallel=False,
        fastmath={"nsz", "ninf", "reassoc", "arcp", "contract", "afn"},
    )
    def _numba_simulate_trade(
        price_times: np.ndarray,
        price_values: np.ndarray,
        weight_times: np.ndarray,
        weight_values: np.ndarray,
        fee: float,
        initial_cash: float,
        min_trade_amount: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate trade with continuous rebalancing.

        Even if the target weights do not change, the portfolio is rebalanced
        at every price timestamp to match the target weights (because price movements
        change the actual portfolio weights).
        """

        n_prices = len(price_times)
        n_weights = len(weight_times)
        n_assets = price_values.shape[1]

        nav_output = np.zeros(n_prices, dtype=np.float64)
        turnover_output = np.zeros(n_prices, dtype=np.float64)

        cash = initial_cash
        shares = np.zeros(n_assets, dtype=np.float64)
        current_target_weights = np.zeros(n_assets, dtype=np.float64)

        w_ptr = 0
        # Flag to check if we have received the first weight signal
        has_received_first_weight = False

        for p_idx in range(n_prices):
            t_p = price_times[p_idx]
            current_prices = price_values[p_idx]

            # 1. Update target weights if new signal exists
            # If multiple weight signals occur before the current price timestamp,
            # we iterate through them and only keep the last one (Last Write Wins).
            while w_ptr < n_weights and weight_times[w_ptr] <= t_p:
                current_target_weights = weight_values[w_ptr]
                w_ptr += 1
                has_received_first_weight = True

            # Fill nan weights with 0
            current_target_weights[np.isnan(current_target_weights)] = 0.0

            # 2. Calculate Pre-Trade NAV (Mark to Market)
            current_invest_value = 0.0
            for i in range(n_assets):
                if not np.isnan(current_prices[i]):
                    current_invest_value += shares[i] * current_prices[i]

            equity_pre_trade = cash + current_invest_value

            # 3. Execute trades (Continuous Rebalance)
            # Once the first weight is received, we check and rebalance at every timestamp.
            total_traded_value_at_t = 0.0

            if has_received_first_weight:
                new_shares = np.copy(shares)

                for i in range(n_assets):
                    price = current_prices[i]

                    if np.isnan(price) or price <= 0:
                        continue

                    # Crucial Step:
                    # Even if current_target_weights haven't changed, price changes cause
                    # equity_pre_trade to change, altering target_val.
                    # Thus, diff != 0, triggering a rebalance.
                    target_val = equity_pre_trade * current_target_weights[i]
                    current_val = shares[i] * price
                    diff = target_val - current_val

                    # Calculate trading volume (Buy + Sell)
                    if abs(diff) >= min_trade_amount:
                        total_traded_value_at_t += abs(diff)

                        if diff > 0:
                            # Buy: Fee is deducted internally (from the cash used to buy)
                            trade_amt = diff
                            fee_amt = trade_amt * fee
                            net_buy_val = trade_amt - fee_amt

                            cash -= trade_amt
                            new_shares[i] += net_buy_val / price

                        elif diff < 0:
                            # Sell: Fee is deducted externally (from the cash received)
                            sell_val = abs(diff)
                            fee_amt = sell_val * fee
                            net_cash_in = sell_val - fee_amt

                            new_shares[i] -= sell_val / price
                            cash += net_cash_in

                shares = new_shares
                # Note: We do NOT set has_received_first_weight = False.
                # We continue to rebalance in the next loop.

            # 4. Calculate Post-Trade NAV
            final_invest_value = 0.0
            for i in range(n_assets):
                if not np.isnan(current_prices[i]):
                    final_invest_value += shares[i] * current_prices[i]

            final_nav = cash + final_invest_value

            nav_output[p_idx] = final_nav
            turnover_output[p_idx] = total_traded_value_at_t

        return nav_output, turnover_output

    nav, turnover = _numba_simulate_trade(
        price_times, price_values, weight_times, weight_values, fee, initial_cash
    )

    return DataFrame(
        p_df.select(
            pl.col(prices.time_col),
            pl.Series("nav", nav),
            pl.Series("turnover", turnover),
        ),
        prices.time_col,
        prices.freq,
        _skip_validate=True,
    )


def portfolio_weights_to_pnl(
    weights: DataFrame, prices: DataFrame, fee: float = 0.0006
) -> DataFrame:
    """
    Calculate profit and loss (PnL) of a portfolio based on price changes and weights.
    Returns a single column DataFrame representing the portfolio's total PnL.

    PnL = (Today_NAV / Yesterday_NAV) - 1
    """
    if not isinstance(weights, DataFrame) or not isinstance(prices, DataFrame):
        raise TypeError("Inputs must be ctalearn DataFrame")

    initial_cash = 100000.0
    result_df = simulate_trade(weights, prices, fee=fee, initial_cash=initial_cash)

    return result_df.select(
        pl.col(prices.time_col),
        ((pl.col("nav") / pl.col("nav").shift(1).fill_null(initial_cash)) - 1.0).alias(
            "pnl"
        ),
    )


def sharpe_ratio(pnl: DataFrame) -> float:
    """
    Calculate the Sharpe ratio.
    Mean(PnL) / Std(PnL) * sqrt(AnnualizeFactor)
    """
    pnl = pnl.copy()

    factor = _get_annualize_factor(pnl)
    target = pl.exclude(pnl.time_col)

    stats = pnl._df.select(mean=target.mean(), std=target.std(ddof=1)).to_dict(
        as_series=False
    )

    mean_val = stats["mean"][0]
    std_val = stats["std"][0]

    if std_val is None or std_val == 0:
        return 0.0

    return float((mean_val / std_val) * np.sqrt(factor))


def sortino_ratio(pnl: DataFrame) -> float:
    """
    Calculate the Sortino ratio.
    Mean(PnL) / Std(PnL < 0) * sqrt(AnnualizeFactor)
    """
    pnl = pnl.copy()
    factor = _get_annualize_factor(pnl)
    col = pl.col(pnl.value_col)

    mean = pnl.select(col.mean()).item()

    downside_std = pnl.select(col.filter(col < 0).pow(2).mean().sqrt()).item()

    if downside_std is None or downside_std == 0:
        return 0.0

    return float((mean / downside_std) * np.sqrt(factor))


def max_drawdown(pnl: DataFrame) -> float:
    """
    Calculate the maximum drawdown.
    Min(CumProd - CumMax(CumSum))
    """
    pnl = pnl.copy()

    target = pl.exclude(pnl.time_col)

    res = (
        pnl._df.select(
            pl.col(pnl.time_col),
            (target + 1.0).cum_prod().alias("cum"),
        )
        .select(pl.col("cum"), pl.col("cum").cum_max().alias("peak"))
        .select(((pl.col("peak") - pl.col("cum")) / pl.col("peak")).max().alias("mdd"))
    )

    mdd = res["mdd"][0]
    return float(mdd) if mdd is not None else 0.0


def annual_return(pnl: DataFrame) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    Formula: (1 + GeoMean)^N - 1
    Which is equivalent to: exp(mean(log(1 + r)) * N) - 1
    """
    pnl = pnl.copy()

    factor = _get_annualize_factor(pnl)
    ann_ret = pnl._df.select(
        (pl.exclude(pnl.time_col) + 1.0).log().mean().mul(factor).exp() - 1
    ).item()

    return float(ann_ret) if ann_ret is not None else 0.0


def calmar_ratio(pnl: DataFrame) -> float:
    """
    Annual Return / Max Drawdown.
    """
    pnl = pnl.copy()

    ann_ret = annual_return(pnl)
    mdd = max_drawdown(pnl)

    if mdd == 0:
        return 0.0
    return ann_ret / mdd


def turnover_ratio(weights: DataFrame, prices: DataFrame) -> float:
    """
    Calculate average turnover ratio.

    Definition:
        Daily Turnover Ratio = (Abs(Buy) + Abs(Sell)) / NAV
        Return Value = Mean(Daily Turnover Ratio)
    """
    result_df = simulate_trade(weights, prices, fee=0.0, initial_cash=100000.0)

    ratio_series = (result_df._df["turnover"] / result_df._df["nav"]).fill_nan(0.0)

    mean_val = ratio_series.mean()
    assert mean_val is None or isinstance(mean_val, (int, float))
    return float(mean_val) if mean_val is not None else 0.0


def margin(weights: DataFrame, prices: DataFrame, fee: float = 0.0006) -> float:
    """
    Average gain or loss per dollar traded.
    Calculated as PnL divided by total dollars traded in a given time period.
    """
    result_df = simulate_trade(weights, prices, fee, initial_cash=100000.0)
    pnl = ts_delta(result_df, 1)
    turnover = ts_delay(result_df, 1)
    margin = pnl.rename({"nav": "margin"}) / turnover.rename({"turnover": "margin"})
    margin = margin.filter(
        ~pl.col("margin").is_null()
        & ~pl.col("margin").is_nan()
        & ~pl.col("margin").is_infinite()
    )
    mean_val = margin._df["margin"].mean()
    assert mean_val is None or isinstance(mean_val, (int, float))
    return float(mean_val) if mean_val is not None else 0.0


def value_at_risk(pnl: DataFrame, alpha: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).

    Parameters
    ----------
    pnl : DataFrame
        Log Returns.
    alpha : float, default=0.95
        Confidence level (e.g., 0.95 means we look at the worst 5%).

    Returns
    -------
    float
        The threshold value (negative float).
        E.g., -0.02 means the worst 5% losses exceed -2%.
    """
    quantile_level = 1 - alpha
    var = pnl.select(
        pl.col(pnl.value_col).quantile(quantile_level, interpolation="linear")
    ).item()

    return float(var) if var is not None else 0.0


def conditional_value_at_risk(pnl: DataFrame, alpha: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVar), also known as Expected Shortfall.
    Formula: E[X | X <= VaR]

    Parameters
    ----------
    pnl : DataFrame
        Log Returns.
    alpha : float, default=0.95
        Confidence level (e.g., 0.95 means we look at the worst 5%).

    Returns
    -------
    float
        The Expected Shortfall.
    """
    var_threshold = value_at_risk(pnl, alpha)

    col = pl.col(pnl.value_col)
    cvar = pnl.select(col.filter(col <= var_threshold).mean()).item()

    return cvar if cvar is not None else 0.0


def skewness(pnl: DataFrame) -> float:
    """
    Calculate Skewness of the PnL distribution
    > 0: Right skewed (Positive skew) - Frequent small losses, rare big gains.
    < 0: Left skewed (Negative skew) - Frequent small gains, rare big losses.
    """
    skew = pnl.select(pl.col(pnl.value_col).skew()).item()

    return skew if skew is not None else 0.0


def kurtosis(pnl: DataFrame, fisher: bool = True) -> float:
    """
    Calculate Kurtosis.

    Parameters
    ----------
    pnl : DataFrame
        PnL in percentage.
    fisher : bool, default=True
        If True (default), calculates 'Excess Kurtosis'.
        If False, calculates 'Raw Kurtosis' (Pearson).
    """
    kurt = pnl.select(pl.col(pnl.value_col).kurtosis(fisher=fisher)).item()

    return kurt if kurt is not None else 0.0


def sr_mdd_ratio(pnl: DataFrame) -> float:
    """
    Sharpe / MDD.
    """
    sr = sharpe_ratio(pnl)
    mdd = max_drawdown(pnl)

    if mdd == 0:
        return 0.0
    return sr / mdd


def longest_drawdown_duration(pnl: DataFrame) -> float:
    """
    Calculate the longest drawdown duration in days.

    Returns:
        float: The maximum duration of a drawdown in days.
    """
    pnl = pnl.copy()

    target = pl.exclude(pnl.time_col)
    equity_col = (target + 1.0).cum_prod().alias("equity")

    df_calc = pnl._df.select(equity_col).with_columns(
        pl.col("equity").cum_max().alias("peak")
    )

    is_in_dd = df_calc.select((pl.col("equity") < pl.col("peak")).alias("in_dd"))

    rle_df = is_in_dd.select(pl.col("in_dd").rle()).unnest("in_dd")

    # Calculate max duration in steps (periods)
    max_duration_steps = (
        rle_df.filter(pl.col("value")).select(pl.col("len").max()).item()
    )

    if max_duration_steps is None:
        return 0.0

    # Convert steps to days using the annualization factor
    factor = _get_annualize_factor(pnl)
    if factor == 0:
        return 0.0

    return float((max_duration_steps / factor) * 365.0)


def winrate(pnl: DataFrame) -> float:
    """
    Calculate winrate: Count(PnL > 0) / Count(PnL != 0)
    """
    pnl = pnl.copy()

    target_col = [c for c in pnl.columns if c != pnl.time_col][0]

    stats = pnl._df.select(
        pos_count=pl.col(target_col).filter(pl.col(target_col) > 0).count(),
        total_count=pl.col(target_col).filter(pl.col(target_col) != 0).count(),
    ).to_dict(as_series=False)

    pos = stats["pos_count"][0]
    total = stats["total_count"][0]

    if total == 0:
        return 0.0
    return float(pos / total)


def fitness(weights: DataFrame, prices: DataFrame, fee: float = 0.0006) -> float:
    pnl = portfolio_weights_to_pnl(weights, prices, fee=fee)
    sr = sharpe_ratio(pnl)
    ar = annual_return(pnl)
    turnover = turnover_ratio(weights, prices)
    return float(sr * np.sqrt(abs(ar) / max(turnover, 0.125)))


def rank_ic(signal: DataFrame, price: DataFrame, horizon: int = 1) -> DataFrame:
    """
    Calculates the Rank Information Coefficient (Rank IC) of an alpha signal.

    Rank IC measures the Spearman correlation between the signal value at time t
    and the future return from time t to t + horizon.

    Parameters
    ----------
    signal : DataFrame
        The alpha signal values.
    price : DataFrame
        The asset prices used to calculate future returns.
    horizon : int, default=1
        The number of periods to look forward for the return calculation.
        Defaults to 1.

    Returns
    -------
    DataFrame
        A DataFrame containing the Rank IC time series.
        Columns: [time_col, 'rank_ic']

    Raises
    ------
    TypeError
        If horizon is not an integer.
    ValueError
        If horizon is less than or equal to 0.
    """
    if not isinstance(signal, DataFrame) or not isinstance(price, DataFrame):
        raise TypeError("Inputs must be ctalearn DataFrame")

    if not isinstance(horizon, int):
        raise TypeError(f"Type of 'horizon' should be int, got {type(horizon)}")

    if horizon <= 0:
        raise ValueError(f"Value of 'horizon' should be greater than 0, got {horizon}")

    signal = signal.copy()
    price = price.copy()

    price_cols = set(price._df.columns) - {price.time_col}
    signal_cols = set(signal._df.columns) - {signal.time_col}
    common_assets = sorted(list(price_cols & signal_cols))

    if not common_assets:
        return price._empty()

    start_time = signal._df[signal.time_col][0]
    end_time = signal._df[signal.time_col][-1]

    future_returns = price / ts_delay(price, horizon) - 1
    future_returns = future_returns.drop_nulls()
    future_returns = future_returns.shift_time(
        -timedelta(seconds=future_returns.freq * horizon)
    )

    future_returns = future_returns.filter(
        (pl.col(price.time_col) >= start_time) & (pl.col(price.time_col) <= end_time)
    )

    if len(future_returns._df) == 0:
        return price._empty()

    # Return Long Table: [time, asset, return]
    return_long = (
        future_returns.select([future_returns.time_col] + common_assets)
        .unpivot(
            index=[future_returns.time_col],
            on=common_assets,
            variable_name="asset",
            value_name="return",
        )
        .sort(["asset", future_returns.time_col])
    )

    # Signal Long Table: [time, asset, signal]
    signal_long = (
        signal.select([signal.time_col] + common_assets)
        .unpivot(
            index=[signal.time_col],
            on=common_assets,
            variable_name="asset",
            value_name="signal",
        )
        .sort(["asset", signal.time_col])
    )

    aligned = return_long._df.join_asof(
        signal_long._df,
        left_on=future_returns.time_col,
        right_on=signal.time_col,
        by="asset",
        strategy="backward",
    )

    aligned = aligned.drop_nulls(subset=["signal", "return"])

    ic_df = (
        aligned.group_by(future_returns.time_col)
        .agg(pl.corr("signal", "return", method="spearman").alias("rank_ic"))
        .sort(price.time_col)
    )

    ic_df = ic_df.filter(
        pl.col("rank_ic").is_not_null() & pl.col("rank_ic").is_not_nan()
    )

    return DataFrame(
        ic_df,
        time_col=price.time_col,
        freq=price.freq,
        _skip_validate=True,
    )


def all_performance_metrics(
    weights: DataFrame,
    prices: DataFrame,
    fee: float = 0.0006,
) -> dict[str, float]:
    pnl = portfolio_weights_to_pnl(weights, prices, fee)

    return {
        "sharpe": sharpe_ratio(pnl),
        "sortino": sortino_ratio(pnl),
        "mdd": max_drawdown(pnl),
        "annual_return": annual_return(pnl),
        "calmar": calmar_ratio(pnl),
        "turnover_ratio": turnover_ratio(weights, prices),
        "sr_mdd_ratio": sr_mdd_ratio(pnl),
        "longest_drawdown_duration": longest_drawdown_duration(pnl),
        "winrate": winrate(pnl),
        "fitness": fitness(weights, prices, fee),
        "margin": margin(weights, prices, fee),
        "skewness": skewness(pnl),
        "kurtosis": kurtosis(pnl, fisher=True),
    }
