import polars as pl

from ctalearn.core.dataframe import DataFrame


def sign(df: DataFrame) -> DataFrame:
    """
    Return the sign of each element in the DataFrame (excluding the time column).

    Returns -1 for negative values, 0 for zero, 1 for positive values, and null
    for missing (null) or NaN values.

    Parameters
    ----------
    df : ctalearn.DataFrame
        Input DataFrame with numeric value columns.

    Returns
    -------
    ctalearn.DataFrame
        New DataFrame with the same shape and time index, containing only -1, 0, 1 or null.

    Raises
    ------
    TypeError
        If `df` is not an instance of the expected DataFrame class.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    df.name = f"sign({df.name})"

    return df.select(pl.col(df.time_col), pl.exclude(df.time_col).sign().fill_nan(None))


def log(df: DataFrame) -> DataFrame:
    """
    Calculate the natural logarithm of input array elements.
    Returns null for non-positive values (<= 0) and for null/NaN input.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    df.name = f"log({df.name})"

    target = pl.exclude(df.time_col)

    # Polars .log() returns -inf for 0 and NaN for negative; emit null instead.
    # NaN input satisfies `target > 0` in Polars, so fill_nan(None) catches it too.
    return df.select(
        pl.col(df.time_col),
        pl.when(target > 0).then(target.log()).otherwise(None).fill_nan(None),
    )


def symmetric_log(df: DataFrame) -> DataFrame:
    """
    Calculate the symmetric logarithm.
    Formula: sign(x) * log(|x| + 1)
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    df.name = f"symmetric_log({df.name})"

    target = pl.exclude(df.time_col)

    return df.select(
        pl.col(df.time_col),
        (target.sign() * (target.abs() + 1).log()).fill_nan(None),
    )


def sqrt(df: DataFrame) -> DataFrame:
    """
    Calculate the square root.
    Returns null for negative values and for null/NaN input.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    df.name = f"sqrt({df.name})"

    target = pl.exclude(df.time_col)

    # Polars .sqrt() returns NaN for negatives; NaN input also satisfies
    # `target >= 0` in Polars, so emit null in both cases.
    return df.select(
        pl.col(df.time_col),
        pl.when(target >= 0).then(target.sqrt()).otherwise(None).fill_nan(None),
    )


def symmetric_sqrt(df: DataFrame) -> DataFrame:
    """
    Calculate the symmetric square root.
    Formula: sign(x) * sqrt(|x|)
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    df.name = f"symmetric_sqrt({df.name})"

    target = pl.exclude(df.time_col)

    return df.select(
        pl.col(df.time_col), (target.sign() * target.abs().sqrt()).fill_nan(None)
    )


def cbrt(df: DataFrame) -> DataFrame:
    """
    Calculate the cubic root.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    df = df.copy()
    df.name = f"cbrt({df.name})"

    target = pl.exclude(df.time_col)

    return df.select(pl.col(df.time_col), target.cbrt().fill_nan(None))


def identity(df: DataFrame) -> DataFrame:
    """
    Return the input DataFrame as is.
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    return df.copy()
