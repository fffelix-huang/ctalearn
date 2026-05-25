from polars.testing import assert_frame_equal

from ctalearn.core.dataframe import DataFrame


def assert_df_equal(df1: DataFrame, df2: DataFrame) -> None:
    """
    Custom assertion helper that sorts columns before comparing.
    This ignores column order but preserves detailed error messages.
    """
    # Sort columns of the underlying Polars DataFrame
    # syntax: df.select(sorted(df.columns))
    d1_sorted = df1._df.select(sorted(df1._df.columns))
    d2_sorted = df2._df.select(sorted(df2._df.columns))

    # Use Polars testing util for rich diff output
    assert_frame_equal(d1_sorted, d2_sorted)
    assert df1.name == df2.name, (
        f"DataFrame name mismatch: '{df1.name}' != '{df2.name}'"
    )
    # freq drives the dense-grid guarantee and the @auto_align fast path, so a
    # regression there must fail the comparison instead of slipping through.
    assert df1.time_col == df2.time_col, (
        f"time_col mismatch: '{df1.time_col}' != '{df2.time_col}'"
    )
    assert df1.freq == df2.freq, f"freq mismatch: {df1.freq} != {df2.freq}"
