from ctalearn.core.dataframe import DataFrame


def train_test_split(
    df: DataFrame, train_size: float = 0.7
) -> tuple[DataFrame, DataFrame]:
    """
    Split the DataFrame into train and test sets sequentially (Time Series Split).

    The split index is ``int(len(df) * train_size)``: the first that many rows go
    to train, the rest to test. Both splits keep the original ``time_col``, ``freq``
    and ``name``.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    train_size : float
        Proportion of the dataset to include in the train split, exclusive
        (0, 1). Note that on very small inputs the floor can make train empty
        (e.g. 1 row with train_size=0.7 -> 0 train rows); test is always
        non-empty since train_size < 1.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        (Train DataFrame, Test DataFrame)
    """
    if not isinstance(df, DataFrame):
        raise TypeError(f"Type of `df` should be DataFrame, got {type(df).__name__}")

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")

    total_rows = len(df._df)
    split_idx = int(total_rows * train_size)

    train_pl = df._df[:split_idx]
    test_pl = df._df[split_idx:]

    df_train = DataFrame(train_pl, df.time_col, df.freq, df.name, _skip_validate=True)

    df_test = DataFrame(test_pl, df.time_col, df.freq, df.name, _skip_validate=True)

    return df_train, df_test
