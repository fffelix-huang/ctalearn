import warnings
from typing import Literal

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

from ctalearn.core.dataframe import DataFrame, _return_empty_aligned


def cs_mean(df: DataFrame, ignore_nan: bool = True) -> DataFrame:
    """
    Calculate the cross-sectional mean for each timestamp (row-wise / axis=1).

    The mean value is broadcast to all original columns, maintaining the same
    column structure as the input DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    ignore_nan : bool, default=True
        If True, ignore NaN/Null values when computing the row mean.
        If False, if any NaN/Null exists in a row, the row mean becomes null.

    Returns
    -------
    DataFrame
        DataFrame with the same column structure as input, where each column
        contains the row-wise mean value (broadcast to all columns).
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    asset_cols = [c for c in df._df.columns if c != df.time_col]

    # Convert to Float64 and map NaN -> Null so Polars horizontal
    # aggregations can treat them consistently.
    base = (
        df._df.with_columns(pl.col(asset_cols).cast(pl.Float64).fill_nan(None))
        if asset_cols
        else df._df
    )

    if not asset_cols:
        return DataFrame(
            base.select(pl.col(df.time_col)),
            df.time_col,
            df.freq,
            f"cs_mean({df.name}, ignore_nan={ignore_nan})",
            _skip_validate=True,
        )

    if ignore_nan:
        mean_val = pl.mean_horizontal(asset_cols)
    else:
        any_null = pl.any_horizontal([pl.col(c).is_null() for c in asset_cols])
        # NaN has already been converted to null above.
        mean_val = (
            pl.when(any_null).then(None).otherwise(pl.mean_horizontal(asset_cols))
        )

    # Broadcast mean to all original columns
    out_pl = base.select(pl.col(df.time_col), *[mean_val.alias(c) for c in asset_cols])
    return DataFrame(
        out_pl,
        df.time_col,
        df.freq,
        f"cs_mean({df.name}, ignore_nan={ignore_nan})",
        _skip_validate=True,
    )


def cs_rank(df: DataFrame, ignore_nan: bool = True) -> DataFrame:
    """
    Rank the values for each timestamp (row-wise).

    Rank is scaled to [0, 1] range where 0 is the lowest and 1 is the highest.
    Break ties by average.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    ignore_nan : bool, default=True
        If True, ignore NaN values and rank only valid values in each row.
        If False, if any Nan exists in a row, the entire row becomes NaN.

    Returns
    -------
    DataFrame
        DataFrame with percentile ranked values (0 to 1).

    Raises
    ------
    TypeError
        If `df` is not a DataFrame.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    target_cols = [c for c in df.columns if c != df.time_col]
    x = df.select(target_cols).cast(pl.Float64).to_numpy()

    _, num_factors = x.shape

    if num_factors == 1:
        # Lone value ranks at the midpoint, but NaN must stay NaN (-> null below).
        result = np.where(np.isnan(x), np.nan, 0.5)
    else:
        nan_policy = "omit" if ignore_nan else "propagate"
        ranks = rankdata(x, method="average", axis=1, nan_policy=nan_policy)

        if ignore_nan:
            valid_count = (~np.isnan(x)).sum(axis=1, keepdims=True).astype(float)

            with np.errstate(divide="ignore", invalid="ignore"):
                result = (ranks - 1) / (valid_count - 1)

            # Handle rows with single valid value
            result = np.where((valid_count == 1) & (~np.isnan(x)), 0.5, result)
            result = np.where(np.isnan(x), np.nan, result)
        else:
            result = (ranks - 1) / (num_factors - 1)
            nan_mask = np.isnan(x).any(axis=1, keepdims=True)
            result = np.where(nan_mask, np.nan, result)

    res_pl = pl.DataFrame(result, schema=target_cols)
    final_pl = pl.concat(
        [df._df.select(df.time_col), res_pl], how="horizontal"
    ).fill_nan(None)

    return DataFrame(
        final_pl, df.time_col, df.freq, f"cs_rank({df.name}, ignore_nan={ignore_nan})"
    )


def cs_zscore(df: DataFrame, ignore_nan: bool = True) -> DataFrame:
    """
    Calculate cross-sectional z-score for each timestamp.

    Z-score is calculated as (value - mean) / std for each row.
    When std is 0 (all values are the same), z-score is set to 0.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    ignore_nan : bool, default=True
        If True, ignore NaN values.
        If False, if any NaN exists in a row, the entire row becomes NaN.

    Returns
    -------
    DataFrame
        DataFrame with z-score normalized values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    target_cols = [c for c in df.columns if c != df.time_col]
    x = df.select(target_cols).cast(pl.Float64).to_numpy()

    if ignore_nan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(x, axis=1, keepdims=True)
            std = np.nanstd(x, axis=1, keepdims=True, ddof=0)
    else:
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True, ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (x - mean) / std

    result = np.where(std == 0, 0, result)

    if ignore_nan:
        result = np.where(np.isnan(x), np.nan, result)
    else:
        nan_mask = np.isnan(x).any(axis=1, keepdims=True)
        result = np.where(nan_mask, np.nan, result)

    res_pl = pl.DataFrame(result, schema=target_cols)
    final_pl = pl.concat(
        [df._df.select(df.time_col), res_pl], how="horizontal"
    ).fill_nan(None)

    return DataFrame(
        final_pl, df.time_col, df.freq, f"cs_zscore({df.name}, ignore_nan={ignore_nan})"
    )


def cs_winsorize(df: DataFrame, std: float, ignore_nan: bool = True) -> DataFrame:
    """
    Winsorize values for each timestamp based on standard deviation.

    Winsorizes x to make sure that all values in x are clipped between the lower and upper limits,
    which are specified as multiple of std.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    std : float
        Number of standard deviations from mean to use as limits.
    ignore_nan : bool, default=True
        If True, ignore NaN values.
        If False, if any NaN exists in a row, the entire row becomes NaN.

    Returns
    -------
    DataFrame
        DataFrame with winsorized values.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    target_cols = [c for c in df.columns if c != df.time_col]
    x = df.select(target_cols).cast(pl.Float64).to_numpy()

    if ignore_nan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(x, axis=1, keepdims=True)
            std_dev = np.nanstd(x, axis=1, keepdims=True, ddof=0)
    else:
        mean = np.mean(x, axis=1, keepdims=True)
        std_dev = np.std(x, axis=1, keepdims=True, ddof=0)

    lower_limit = mean - std * std_dev
    upper_limit = mean + std * std_dev
    result = np.clip(x, lower_limit, upper_limit)

    if ignore_nan:
        result = np.where(np.isnan(x), np.nan, result)
    else:
        nan_mask = np.isnan(x).any(axis=1, keepdims=True)
        result = np.where(nan_mask, np.nan, result)

    res_pl = pl.DataFrame(result, schema=target_cols)
    final_pl = pl.concat(
        [df._df.select(df.time_col), res_pl], how="horizontal"
    ).fill_nan(None)

    return DataFrame(
        final_pl,
        df.time_col,
        df.freq,
        f"cs_winsorize({df.name}, std={std}, ignore_nan={ignore_nan})",
    )


def vector_neut(x: DataFrame, y: DataFrame) -> DataFrame:
    """
    Perform cross-sectional vector neutralization.
    Formula: x - (dot(x, y) / dot(y, y)) * y

    For given vectors x and y, it finds a new vector x' (output) such that x' is orthogonal to y.

    Parameters
    ----------
    x : DataFrame
        Input DataFrame X
    y : DataFrame
        Input DataFrame Y (Target)

    Returns
    -------
    DataFrame
        DataFrame with neutralized values.
    """
    if not isinstance(x, DataFrame):
        raise TypeError(f"Type of `x` should be DataFrame, got {type(x).__name__}")

    if not isinstance(y, DataFrame):
        raise TypeError(f"Type of `y` should be DataFrame, got {type(y).__name__}")

    x = x.copy()
    y = y.copy()

    x, y = x.align(y, how="inner")

    # align() only aligns timestamps, not columns. The projection reads y[c] for
    # every column c of x, so a column present in one but not the other would
    # otherwise surface as an opaque KeyError.
    x_cols = set(x._df.columns) - {x.time_col}
    y_cols = set(y._df.columns) - {y.time_col}
    if x_cols != y_cols:
        raise ValueError("`x` and `y` must have the same asset columns")

    value_cols = sorted(list(x_cols))

    xy_dot = (x * y).select(
        pl.col(x.time_col), pl.sum_horizontal(pl.exclude(x.time_col)).alias("dot")
    )
    yy_dot_exprs = [
        pl.when(x.get_column(c).is_not_null())
        .then(pl.col(c) * pl.col(c))
        .otherwise(None)
        for c in value_cols
    ]
    yy_dot = y.select(pl.col(y.time_col), pl.sum_horizontal(yy_dot_exprs).alias("dot"))
    proj_coeff = xy_dot / (yy_dot + 1e-12)

    df = x.select(
        pl.col(x.time_col),
        *[
            (pl.col(c) - proj_coeff.get_column("dot") * y.get_column(c)).alias(c)
            for c in value_cols
        ],
    )
    df.name = f"vector_neut({x.name}, {y.name})"
    return df


def regression_neut(
    target: DataFrame,
    neut_factors: DataFrame | list[DataFrame],
    ridge_alpha: float = 1e-8,
) -> DataFrame:
    """
    Fully vectorized cross-sectional regression using Batch Linear Algebra.
    No explicit Python loops over timestamps.
    """
    if isinstance(neut_factors, DataFrame):
        neut_factors = [neut_factors]

    neut_names = ",".join([f.name for f in neut_factors])
    new_name = f"regression_neut({target.name}, [{neut_names}])"

    # 1. Align DataFrames
    aligned_target = target
    for f in neut_factors:
        aligned_target, _ = aligned_target.align(f, how="inner")

    if len(aligned_target._df) == 0:
        cols = sorted(list(set(target._df.columns) - {target.time_col}))
        return _return_empty_aligned(target, target, lambda x, y, z: x, cols, 1)

    aligned_neuts: list[DataFrame] = []
    for f in neut_factors:
        _, f_aligned = aligned_target.align(f, how="inner")
        aligned_neuts.append(f_aligned)

    time_col = target.time_col
    common_cols = set(aligned_target._df.columns) - {time_col}
    for f in aligned_neuts:
        common_cols &= set(f._df.columns) - {f.time_col}

    if not common_cols:
        raise ValueError("No common columns (assets) found.")

    sorted_cols = sorted(list(common_cols))

    # Y: (T, N)
    Y_mat = aligned_target._df.select(sorted_cols).to_numpy().astype(np.float64)
    # X: (T, N, K)
    X_list = [
        f._df.select(sorted_cols).to_numpy().astype(np.float64) for f in aligned_neuts
    ]
    X_mat = np.stack(X_list, axis=-1)

    # 2. Batch Linear Regression
    T, N = Y_mat.shape
    K = len(aligned_neuts)

    nan_mask = np.isnan(Y_mat) | np.isnan(X_mat).any(axis=2)
    valid_mask = ~nan_mask

    Y_filled = np.nan_to_num(Y_mat, nan=0.0)
    X_filled = np.nan_to_num(X_mat, nan=0.0)

    intercept = valid_mask.astype(float)  # (T, N)
    X_design = np.concatenate([intercept[:, :, None], X_filled], axis=2)

    X_design = X_design * valid_mask[:, :, None]

    Xt = X_design.transpose(0, 2, 1)
    XtX = Xt @ X_design

    Y_target = (Y_filled * valid_mask)[:, :, None]
    XtY = Xt @ Y_target  # (T, K+1, 1)

    if ridge_alpha > 0:
        I_ridge = np.eye(K + 1)
        I_ridge[0, 0] = 0
        XtX += ridge_alpha * I_ridge

    n_valid_counts = valid_mask.sum(axis=1)
    bad_dof_mask = n_valid_counts < (K + 2)

    if bad_dof_mask.any():
        XtX[bad_dof_mask] = np.eye(K + 1)
        XtY[bad_dof_mask] = 0

    try:
        betas = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        return DataFrame(
            pl.concat(
                [
                    aligned_target._df.select(time_col),
                    pl.DataFrame(np.full((T, N), np.nan), schema=sorted_cols),
                ],
                how="horizontal",
            ),
            time_col,
            aligned_target.freq,
            name=new_name,
            _skip_validate=True,
        )

    preds = X_design @ betas
    residuals = Y_target - preds
    residuals = residuals.squeeze(-1)  # (T, N)

    residuals[~valid_mask] = np.nan
    residuals[bad_dof_mask, :] = np.nan

    # 3. Reconstruct DataFrame
    res_df_pl = pl.DataFrame(residuals, schema=sorted_cols, orient="row")
    final_pl = pl.concat(
        [aligned_target._df.select(time_col), res_df_pl], how="horizontal"
    )

    return DataFrame(
        final_pl, time_col, freq=aligned_target.freq, name=new_name, _skip_validate=True
    )


def process_alpha_weights(
    alpha_signals: DataFrame, neutralize: bool = True
) -> DataFrame:
    """
    Processes raw alpha signal matrix by applying optional neutralization
    and subsequent normalization to each row independently.

    This function transforms a matrix of raw alpha scores into a matrix of
    portfolio weight vectors, ensuring NaN values are converted to zero weights.

    Parameters
    ----------
    alpha_signals : DataFrame
        A DataFrame of raw alpha values, where each row represents a
        different period or asset universe.
    neutralize : bool, default=True
        If True, each row (signal vector) is market-neutralized by
        subtracting its mean (calculated from non-NaN values), resulting
        in row sums of zero (Long/Short strategy).
        If False, no mean subtraction is performed.

    Returns
    -------
    DataFrame
        For each row, the absolute sum of non-zero weights is scaled to 1.0.
        All input NaN values are set to 0.0, indicating no capital allocation.
    """
    if not isinstance(alpha_signals, DataFrame):
        raise TypeError(
            f"Type of `df` should be DataFrame, got {type(alpha_signals).__name__}"
        )

    df = alpha_signals._df
    asset_cols = [col for col in df.columns if col != alpha_signals.time_col]
    asset_exprs = pl.col(asset_cols)

    # 1. Neutralization
    if neutralize:
        row_means = pl.mean_horizontal(asset_cols)
        df = df.with_columns(
            (pl.col(name) - row_means).alias(name) for name in asset_cols
        )

    # 2. Normalization
    row_abs_sums = pl.sum_horizontal(asset_exprs.abs())
    df = df.with_columns(
        (pl.col(name) / row_abs_sums).fill_nan(0.0).fill_null(0.0).alias(name)
        for name in asset_cols
    )

    return DataFrame(
        df, alpha_signals.time_col, freq=alpha_signals.freq, _skip_validate=True
    )


def cs_pca(
    df: DataFrame,
    window: int,
    n_components: int = 1,
    output: Literal["scores", "loadings", "explained_variance"] = "scores",
) -> "DataFrame | tuple[DataFrame, ...]":
    """
    Perform high-performance vectorized rolling cross-sectional PCA with Sign Correction.

    Includes Max Loading Sign Correction to ensure consistent signal direction.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with time column and feature columns (symbols).
    window : int
        Rolling window size. Must be > 1.
    n_components : int, default=1
        Number of principal components to compute.
    output : {"scores", "loadings", "explained_variance"}, default="scores"
        Type of output to return:
        - "scores": Principal component scores
        - "loadings": Principal component loadings (eigenvectors)
        - "explained_variance": Explained variance ratio for each component

    Returns
    -------
    DataFrame or tuple[DataFrame, ...]
        - If n_components == 1: Returns a single DataFrame
        - If n_components > 1: Returns a tuple of DataFrames (pc1_df, pc2_df, ...)

        For all output types, each DataFrame has the same column structure as the input:
        - "loadings": Each column contains the loading value for that feature/symbol
        - "scores": The scalar score is broadcast to all original columns
        - "explained_variance": The scalar variance ratio is broadcast to all original columns

    Raises
    ------
    TypeError
        If `df` is not a DataFrame.
    ValueError
        If `window` <= 1, `n_components` <= 0, `n_components` > number of features,
        or `output` is not a valid option.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if window <= 1:
        raise ValueError(f"`window` must be an integer > 1, got {window}")

    if n_components <= 0:
        raise ValueError(f"`n_components` must be a positive integer")

    valid_outputs = ("scores", "loadings", "explained_variance")
    if output not in valid_outputs:
        raise ValueError(f"`output` must be one of {valid_outputs}, got '{output}'")

    # 1. Prepare data
    df_copy = df.copy()
    target_cols = [c for c in df_copy.columns if c != df_copy.time_col]
    num_features = len(target_cols)

    if n_components > num_features:
        raise ValueError(
            f"`n_components` ({n_components}) exceeds number of features ({num_features})"
        )

    # To Numpy matrix
    X = df_copy.select(target_cols).cast(pl.Float64).to_numpy()
    n_rows, n_cols = X.shape

    def _build_result_dataframes(
        result_data_list: list[np.ndarray],
        target_cols: list[str],
        time_df: pl.DataFrame,
        time_col: str,
        freq: int | None,
        df_name: str,
        output_type: str,
    ) -> "DataFrame | tuple[DataFrame, ...]":
        """Build result DataFrames from computed data arrays."""
        result_dfs = []
        for pc_idx, data in enumerate(result_data_list):
            res_pl = pl.DataFrame(data, schema=target_cols)
            final_pl = pl.concat([time_df, res_pl], how="horizontal").fill_nan(None)
            pc_name = f"cs_pca({df_name}, {window}, {n_components}, output='{output_type}')_PC{pc_idx + 1}"
            result_dfs.append(
                DataFrame(final_pl, time_col, freq, pc_name, _skip_validate=True)
            )
        if len(result_dfs) == 1:
            return result_dfs[0]
        return tuple(result_dfs)

    # Not enough rows, only pad result with nan
    if n_rows < window:
        nan_data = np.full((n_rows, num_features), np.nan)
        result_data_list = [nan_data for _ in range(n_components)]
        return _build_result_dataframes(
            result_data_list,
            target_cols,
            df._df.select(df.time_col),
            df.time_col,
            df.freq,
            df.name,
            output,
        )

    # 2. Sliding window
    X_windows = sliding_window_view(X, window_shape=(window, n_cols)).reshape(
        -1, window, n_cols
    )

    # 3. Vectorized Z-Score normalization
    means = np.mean(X_windows, axis=1, keepdims=True)
    stds = np.std(X_windows, axis=1, ddof=0, keepdims=True)
    stds = np.where(stds < 1e-12, 1.0, stds)
    X_norm = (X_windows - means) / stds

    # 4. Vectorized covariance
    cov_matrices = (X_norm.transpose(0, 2, 1) @ X_norm) / (window - 1)

    # 5. Eigenvalues & vectors
    try:
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrices)
    except np.linalg.LinAlgError:
        eig_vals = np.full((len(cov_matrices), n_cols), np.nan)
        eig_vecs = np.full((len(cov_matrices), n_cols, n_cols), np.nan)

    eig_vals = eig_vals[:, ::-1]
    eig_vecs = eig_vecs[:, :, ::-1]

    top_eig_vals = eig_vals[:, :n_components]  # (N_win, k)
    top_eig_vecs = eig_vecs[:, :, :n_components]  # (N_win, F, k)

    # -- Sign correction
    if output in ["scores", "loadings"]:
        max_abs_idx = np.argmax(np.abs(top_eig_vecs), axis=1)  # (N_win, k)
        n_win_range = np.arange(top_eig_vecs.shape[0])[:, None]  # (N_win, 1)
        k_range = np.arange(n_components)[None, :]  # (1, k)
        max_vals = top_eig_vecs[n_win_range, max_abs_idx, k_range]
        signs = np.sign(max_vals)
        signs[signs == 0] = 1.0
        top_eig_vecs = top_eig_vecs * signs[:, np.newaxis, :]

    # Pad
    pad_len = window - 1

    # -- Output: Build result data list for each PC
    result_data_list = []

    if output == "scores":
        current_obs = X_norm[:, -1, :][:, np.newaxis, :]
        scores = (current_obs @ top_eig_vecs).squeeze(axis=1)  # (N_win, k)
        # Broadcast each scalar score to all columns
        for pc_idx in range(n_components):
            score_col = scores[:, pc_idx : pc_idx + 1]  # (N_win, 1)
            broadcasted = np.broadcast_to(score_col, (scores.shape[0], num_features))
            padded = np.vstack([np.full((pad_len, num_features), np.nan), broadcasted])
            result_data_list.append(padded)

    elif output == "loadings":
        # top_eig_vecs: (N_win, F, k) - each column is already the loading for that feature
        for pc_idx in range(n_components):
            loadings = top_eig_vecs[:, :, pc_idx]  # (N_win, F)
            padded = np.vstack([np.full((pad_len, num_features), np.nan), loadings])
            result_data_list.append(padded)

    else:  # explained_variance
        total_var = eig_vals.sum(axis=1, keepdims=True)
        total_var[total_var == 0] = 1.0
        explained_ratio = top_eig_vals / total_var  # (N_win, k)
        # Broadcast each scalar variance ratio to all columns
        for pc_idx in range(n_components):
            var_col = explained_ratio[:, pc_idx : pc_idx + 1]  # (N_win, 1)
            broadcasted = np.broadcast_to(
                var_col, (explained_ratio.shape[0], num_features)
            )
            padded = np.vstack([np.full((pad_len, num_features), np.nan), broadcasted])
            result_data_list.append(padded)

    return _build_result_dataframes(
        result_data_list,
        target_cols,
        df._df.select(df.time_col),
        df.time_col,
        df.freq,
        df.name,
        output,
    )
