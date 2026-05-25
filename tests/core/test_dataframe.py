from datetime import datetime, timedelta

import polars as pl
import pytest

from ctalearn.core.dataframe import DataFrame


class TestDataFrame:
    @pytest.fixture
    def df_a(self) -> DataFrame:
        """
        Base DF: Freq 2s, Range 10:00:00 - 10:00:10
        Values: 10, 20, 30, 40, 50, 60
        """
        dates = pl.datetime_range(
            start=datetime(2023, 1, 1, 10, 0, 0),
            end=datetime(2023, 1, 1, 10, 0, 10),
            interval="2s",
            eager=True,
        )
        return DataFrame(
            pl.DataFrame(
                {
                    "ts": dates,
                    "val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                    "extra_a": [1] * 6,
                }
            ),
            "ts",
            freq=2,
            name="df_a",
        )

    @pytest.fixture
    def df_b(self) -> DataFrame:
        """
        Offset DF: Freq 3s, Range 10:00:03 - 10:00:15 (Different Time Col Name)
        Values: 5, 15, 25, 35, 45
        """
        dates = pl.datetime_range(
            start=datetime(2023, 1, 1, 10, 0, 3),
            end=datetime(2023, 1, 1, 10, 0, 15),
            interval="3s",
            eager=True,
        )
        return DataFrame(
            pl.DataFrame(
                {
                    "date": dates,
                    "val": [5.0, 15.0, 25.0, 35.0, 45.0],
                    "extra_b": [2] * 5,
                }
            ),
            "date",
            freq=3,
            name="df_b",
        )

    def test_value_col_single(self, df_b: DataFrame) -> None:
        # One non-time value column -> returned regardless of its name.
        single = df_b.select(["date", "val"])
        assert single.value_col == "val"

    def test_value_col_multiple_raises(self, df_a: DataFrame) -> None:
        # df_a has two value columns (val, extra_a) -> ambiguous.
        with pytest.raises(ValueError, match="single value column"):
            _ = df_a.value_col

    def test_value_col_none_raises(self, df_a: DataFrame) -> None:
        time_only = df_a.select(["ts"])
        with pytest.raises(ValueError, match="single value column"):
            _ = time_only.value_col

    def test_calculate_freq_regular(self) -> None:
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 10),
                    datetime(2023, 1, 1, 10, 0, 20),
                ],
                "val": [1, 2, 3],
            }
        )
        d = DataFrame(df, "ts")
        assert d.freq == 10

    def test_calculate_freq_gcd_not_min(self) -> None:
        # gaps 120s & 180s -> base period is gcd=60, not min=120
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 2, 0),
                    datetime(2023, 1, 1, 10, 5, 0),
                ],
                "val": [1.0, 2.0, 3.0],
            }
        )
        d = DataFrame(df, "ts")
        assert d.freq == 60

    def test_calculate_freq_irregular_fill(self) -> None:
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2023, 1, 1, 10, 0, 0),
                    datetime(2023, 1, 1, 10, 0, 10),
                    datetime(2023, 1, 1, 10, 0, 15),
                ],
                "val": [1, 2, 3],
            }
        )
        result = DataFrame(df, "ts")
        assert result._df[result.time_col][0] == datetime(2023, 1, 1, 10, 0, 0)
        assert result._df[result.time_col][1] == datetime(2023, 1, 1, 10, 0, 5)
        assert result._df[result.time_col][2] == datetime(2023, 1, 1, 10, 0, 10)
        assert result._df[result.time_col][3] == datetime(2023, 1, 1, 10, 0, 15)
        assert result.freq == 5

    def test_calculate_freq_duplicates_raises(self) -> None:
        df = pl.DataFrame(
            {
                "ts": [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 10, 0, 0)],
                "val": [1, 2],
            }
        )
        with pytest.raises(ValueError, match="Duplicate timestamps detected"):
            DataFrame(df, "ts")

    def test_calculate_freq_single_row(self) -> None:
        df = pl.DataFrame({"ts": [datetime(2023, 1, 1, 10, 0, 0)], "val": [1]})
        d = DataFrame(df, "ts")
        assert d.freq is None

    def test_ensure_dense_fills_gaps(self) -> None:
        df = pl.DataFrame(
            {
                "ts": [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 10, 0, 20)],
                "val": [10, 20],
            }
        )
        d = DataFrame(df, "ts", freq=10)

        assert len(d._df) == 3
        val_10 = (
            d.filter(pl.col("ts") == datetime(2023, 1, 1, 10, 0, 10))
            .select("val")
            .item()
        )
        assert val_10 == 10

    def test_ensure_dense_maintains_explicit_nulls(self) -> None:
        df_raw = pl.DataFrame(
            {
                "ts": [
                    datetime(2023, 1, 1, 1),
                    datetime(2023, 1, 1, 2),
                    datetime(2023, 1, 1, 3),
                ],
                "val": [1, None, 3],
            }
        )

        d = DataFrame(df_raw, "ts", freq=600)

        # within one bar of 1:00 -> forward-filled
        val_110 = (
            d.filter(pl.col("ts") == datetime(2023, 1, 1, 1, 10)).select("val").item()
        )
        assert val_110 == 1

        # beyond one bar (gap) -> null, not fabricated
        val_150 = (
            d.filter(pl.col("ts") == datetime(2023, 1, 1, 1, 50)).select("val").item()
        )
        assert val_150 is None

        val_200 = (
            d.filter(pl.col("ts") == datetime(2023, 1, 1, 2, 0)).select("val").item()
        )
        assert val_200 is None

        val_210 = (
            d.filter(pl.col("ts") == datetime(2023, 1, 1, 2, 10)).select("val").item()
        )
        assert val_210 is None

    def test_ensure_dense_bounds_large_gaps(self) -> None:
        # Bug 8a: gaps larger than one bar must be null, not backfilled.
        base = datetime(2023, 1, 1)
        df = pl.DataFrame(
            {
                "ts": [
                    base,
                    base + timedelta(seconds=60),
                    base + timedelta(seconds=600),
                ],
                "val": [10.0, 20.0, 99.0],
            }
        )
        d = DataFrame(df, "ts")  # freq = gcd(60, 540) = 60
        assert d.freq == 60

        def val_at(sec: int) -> object:
            return (
                d.filter(pl.col("ts") == base + timedelta(seconds=sec))
                .select("val")
                .item()
            )

        assert val_at(60) == 20.0  # real
        assert val_at(120) == 20.0  # one bar after real -> filled
        assert val_at(180) is None  # two bars into gap -> null
        assert val_at(540) is None
        assert val_at(600) == 99.0  # real

    def test_ensure_dense_nan_barrier(self) -> None:
        # A float NaN at a real point is preserved and is not skipped over to
        # forward-fill an earlier non-NaN value; fill stays bounded to one bar.
        base = datetime(2023, 1, 1)
        df = pl.DataFrame(
            {
                "ts": [base + timedelta(seconds=s) for s in (0, 30, 60)],
                "val": [1.0, float("nan"), 3.0],
            }
        )
        d = DataFrame(df, "ts", freq=10)

        def val_at(sec: int) -> object:
            return (
                d.filter(pl.col("ts") == base + timedelta(seconds=sec))
                .select("val")
                .item()
            )

        assert val_at(10) == 1.0  # one bar after real 1.0 -> filled
        assert val_at(20) is None  # two bars -> null, NOT 1.0
        assert val_at(30) != val_at(30) or val_at(30) is None  # NaN preserved
        assert val_at(40) != val_at(40)  # NaN carried one bar (still NaN, not 1.0)
        assert val_at(50) is None  # beyond one bar -> null

    def test_align_different_freq_null_barrier(self) -> None:
        # In the different-freq align path, a null at a real source point stays
        # null (not overwritten by an earlier value), and fill is bounded.
        base = datetime(2023, 1, 1)
        a = DataFrame(
            pl.DataFrame(
                {
                    "t": [base + timedelta(seconds=s) for s in (0, 2, 4, 6)],
                    "v": [10.0, None, 30.0, 40.0],
                }
            ),
            "t",
            freq=2,
        )
        b = DataFrame(
            pl.DataFrame(
                {
                    "t": [base + timedelta(seconds=s) for s in (0, 3, 6)],
                    "v": [5.0, 15.0, 25.0],
                }
            ),
            "t",
            freq=3,
        )
        a2, _ = a.align(b, how="inner")  # grid: gcd=1, 0..6

        def a_at(sec: int) -> object:
            return (
                a2.filter(pl.col("t") == base + timedelta(seconds=sec))
                .select("v")
                .item()
            )

        assert a_at(0) == 10.0
        assert a_at(1) == 10.0  # filled one bar (tol = 2-1 = 1)
        assert a_at(2) is None  # real null preserved, not 10.0
        assert a_at(3) is None  # backward match is the null at @2
        assert a_at(4) == 30.0

    def test_align_outer_bounds_tail_gap(self) -> None:
        # how="outer" must bound the fill exactly like how="inner": grid slots past
        # the shorter frame's last point stay null instead of being fabricated.
        base = datetime(2023, 1, 1)
        a = DataFrame(
            pl.DataFrame(
                {
                    "t": [base + timedelta(seconds=s) for s in (0, 1, 2)],
                    "v": [1.0, 2.0, 3.0],
                }
            ),
            "t",
            freq=1,
        )
        b = DataFrame(
            pl.DataFrame(
                {
                    "t": [base + timedelta(seconds=s) for s in (0, 1, 2, 3, 4, 5)],
                    "v": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                }
            ),
            "t",
            freq=1,
        )
        a2, b2 = a.align(b, how="outer")

        def a_at(sec: int) -> object:
            return (
                a2.filter(pl.col("t") == base + timedelta(seconds=sec))
                .select("v")
                .item()
            )

        assert a_at(2) == 3.0
        assert a_at(3) is None  # was fabricated as 3.0 with tolerance=None
        assert a_at(5) is None
        # The longer frame is unaffected.
        assert b2._df["v"].to_list() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    def test_forced_freq_off_grid_raises(self) -> None:
        # Bug 8b: forced freq that doesn't divide the spacing must not silently relabel.
        base = datetime(2023, 1, 1)
        df = pl.DataFrame(
            {
                "ts": [base + timedelta(seconds=s) for s in (0, 10, 15, 30)],
                "val": [1.0, 2.0, 77.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="grid"):
            DataFrame(df, "ts", freq=10)

    def test_skip_validate_bypasses_sort(self) -> None:
        df_unsorted = pl.DataFrame(
            {
                "ts": [
                    datetime(2023, 1, 1, 10, 0, 20),
                    datetime(2023, 1, 1, 10, 0, 10),
                ],
                "val": [2, 1],
            }
        )

        d = DataFrame(df_unsorted, "ts", freq=10, _skip_validate=True)

        assert d._df["ts"][0] == datetime(2023, 1, 1, 10, 0, 20)

    def test_shift_time(self, df_a: DataFrame) -> None:
        df = df_a.shift_time(timedelta(hours=2))

        for i in range(len(df._df)):
            assert df._df[df.time_col][i] == df_a._df[df_a.time_col][i] + timedelta(
                hours=2
            )

        assert df.time_col == df_a.time_col
        assert df.freq == df_a.freq
        assert len(df._df) == len(df_a._df)
        assert df._df.columns == df_a._df.columns
        assert df.name == "df_a.shift_time(datetime.timedelta(seconds=7200))"

    def test_align_forward_fill(self, df_a: DataFrame, df_b: DataFrame) -> None:
        df_a, df_b = df_a.align(df_b, how="inner")

        assert df_a.freq == 1
        assert len(df_a._df) == 8
        assert df_a._df["val"][0] == 20.0

    def test_align_different_freq_gcd(self, df_a: DataFrame, df_b: DataFrame) -> None:
        res = df_a + df_b

        assert res.freq == 1
        assert len(res._df) == 8
        assert res._df["ts"][0] == datetime(2023, 1, 1, 10, 0, 3)
        assert res._df["ts"][-1] == datetime(2023, 1, 1, 10, 0, 10)
        assert "val" in res._df.columns

    def test_align_empty_intersection(self) -> None:
        df_a = DataFrame(
            pl.DataFrame({"ts": [datetime(2023, 1, 1, 0, 0, 0)], "val": [1]}),
            "ts",
            freq=1,
        )

        df_b = DataFrame(
            pl.DataFrame({"ts": [datetime(2023, 1, 1, 1, 0, 0)], "val": [1]}),
            "ts",
            freq=1,
        )

        res = df_a + df_b
        assert len(res._df) == 0
        assert res.freq == 1

    def test_align_empty_operand_no_crash(self) -> None:
        """Bug 2: arithmetic w/ empty-but-freq'd operand must not raise IndexError."""
        dates = pl.datetime_range(
            start=datetime(2023, 1, 1, 10, 0, 0),
            end=datetime(2023, 1, 1, 10, 0, 10),
            interval="2s",
            eager=True,
        )
        d = DataFrame(
            pl.DataFrame({"ts": dates, "val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}),
            "ts",
            freq=2,
        )
        # filter to 0 rows; __getattr__ proxy keeps freq=2
        empty = d.filter(pl.col("ts") < datetime(2023, 1, 1, 10, 0, 0))
        assert len(empty._df) == 0
        assert empty.freq == 2

        res = empty + d
        assert len(res._df) == 0
        assert "val" in res._df.columns

    def test_align_empty_intersection_preserves_freq(self) -> None:
        """Bug 3: empty intersection keeps aligned freq, not hardcoded 1."""
        a = DataFrame(
            pl.DataFrame(
                {
                    "ts": pl.datetime_range(
                        datetime(2023, 1, 1, 0, 0, 0),
                        datetime(2023, 1, 1, 0, 4, 0),
                        interval="60s",
                        eager=True,
                    ),
                    "val": [1.0] * 5,
                }
            ),
            "ts",
            freq=60,
        )
        b = DataFrame(
            pl.DataFrame(
                {
                    "ts": pl.datetime_range(
                        datetime(2023, 1, 1, 1, 0, 0),
                        datetime(2023, 1, 1, 1, 4, 0),
                        interval="60s",
                        eager=True,
                    ),
                    "val": [1.0] * 5,
                }
            ),
            "ts",
            freq=60,
        )
        res = a + b
        assert len(res._df) == 0
        assert res.freq == 60

    def test_rename_basic(self, df_a: DataFrame) -> None:
        df = df_a.rename({"val": "value"})
        assert "value" in df._df.columns
        assert "val" not in df._df.columns
        assert "ts" in df._df.columns
        assert df.name == "df_a.rename({'val': 'value'})"

    def test_rename_inplace(self, df_a: DataFrame) -> None:
        df_a.rename({"val": "value"}, inplace=True)
        assert "value" in df_a._df.columns
        assert "val" not in df_a._df.columns
        assert "ts" in df_a._df.columns
        assert df_a.name == "df_a.rename({'val': 'value'})"

    def test_concat_inner(self, df_a: DataFrame, df_b: DataFrame) -> None:
        df_a.rename({"val": "val_a"}, inplace=True)
        df = df_a.concat(df_b, how="inner")

        assert len(df._df) == 8
        assert "val_a" in df._df.columns
        assert "val" in df._df.columns
        assert df.name == "df_a.rename({'val': 'val_a'}).concat(df_b, how='inner')"

    def test_concat_outer(self, df_a: DataFrame, df_b: DataFrame) -> None:
        df_a.rename({"val": "val_a"}, inplace=True)
        df = df_a.concat(df_b, how="outer")

        assert len(df._df) == 16
        assert "val_a" in df._df.columns
        assert "val" in df._df.columns
        assert df.name == "df_a.rename({'val': 'val_a'}).concat(df_b, how='outer')"

    # --- Alignment & Operation Tests ---

    def test_subtraction_logic(self, df_a: DataFrame, df_b: DataFrame) -> None:
        res = df_a - df_b

        # 1. Check Return Type & Attributes
        assert isinstance(res, DataFrame)
        assert res.time_col == "ts"  # Should follow df_a (self)
        assert res.freq == 1
        assert "date" not in res._df.columns
        assert res.name == "(df_a - df_b)"

        # 2. Check Dimensions
        assert len(res._df) == 8  # 03, 04, 05, 06, 07, 08, 09, 10

        # 3. Check Value Calculation at specific timestamp
        # Timestamp: 10:00:04
        # A (2s): 02=20, 04=30. At 04 is 30.
        # B (3s): 03=5, 06=15. At 04 is ffill(03) = 5.
        # Expected: 30 - 5 = 25
        val_04 = (
            res.filter(pl.col("ts") == datetime(2023, 1, 1, 10, 0, 4))
            .select("val")
            .item()
        )
        assert val_04 == 25.0

        # Timestamp: 10:00:05
        # A (2s): 04=30. At 05 is ffill(04) = 30.
        # B (3s): 03=5. At 05 is ffill(03) = 5.
        # Expected: 30 - 5 = 25
        val_05 = (
            res.filter(pl.col("ts") == datetime(2023, 1, 1, 10, 0, 5))
            .select("val")
            .item()
        )
        assert val_05 == 25.0

    def test_no_overlap(self, df_a: DataFrame, df_b: DataFrame) -> None:
        """Test case with absolutely no time overlap."""
        # Shift B by 1 hour
        df_late = DataFrame(
            df_b._df.with_columns(pl.col("date") + pl.duration(hours=1)), "date", freq=3
        )

        res = df_a + df_late

        # Should return empty DataFrame but preserve Schema
        assert len(res._df) == 0
        assert "val" in res._df.columns
        assert "ts" in res._df.columns

    def test_no_common_columns(self, df_a: DataFrame, df_b: DataFrame) -> None:
        """Test case with no common columns."""
        # Rename 'val' in B so there are no common data columns
        df_unique = DataFrame(df_b._df.rename({"val": "val_b"}), "date", freq=3)

        res = df_a * df_unique

        # Schema should only contain time_col (since only time_col is common, but excluded from ops)
        assert res._df.columns == ["ts"]

    # --- Polars Proxy Tests ---

    def test_polars_native_behavior(self, df_a: DataFrame) -> None:
        """
        Test proxying of Polars native operations (__getattr__).
        Includes Filter, Select.
        """
        # 1. Test Filter
        # Originally 6 rows (10, 20, 30, 40, 50, 60) -> >30 leaves 3 rows
        filtered = df_a.filter(pl.col("val") > 30)
        assert isinstance(filtered, DataFrame)
        assert len(filtered._df) == 3
        assert filtered._df["val"].min() == 40.0

        # 2. Test Select
        selected = df_a.select(["ts", "val"])
        assert isinstance(selected, DataFrame)
        assert "extra_a" not in selected._df.columns

    def test_chaining_operations(self, df_a: DataFrame) -> None:
        """Test chaining operations."""
        # (Filter -> Select) Should still return DataFrame
        res = df_a.filter(pl.col("val") > 10).select("val")
        assert isinstance(res, DataFrame)
        assert len(res._df) == 5
        # Note: if select doesn't pick time_col, it grabs the first column as time_col (based on getattr impl)
        # But our getattr proxy passes the original time_col ("ts") even if not in result,
        # which might be risky if dropped.
        # Assuming usage keeps time_col or we rely on internal _df behavior.
        # In current impl, getattr returns DataFrame(result, self.time_col, self.freq)
        assert res.time_col == "ts"

    # --- Scalar Operation Tests ---

    def test_add_scalar(self, df_a: DataFrame) -> None:
        res = df_a + 5.0
        assert isinstance(res, DataFrame)
        # 10 + 5 = 15
        assert res._df["val"][0] == 15.0
        # Check time column is untouched
        assert res._df["ts"][0] == df_a._df["ts"][0]
        assert res.name == "(df_a + 5.0)"

    def test_sub_scalar(self, df_a: DataFrame) -> None:
        res = df_a - 1.0
        assert res._df["val"][0] == 9.0
        assert res.name == "(df_a - 1.0)"

    def test_mul_scalar(self, df_a: DataFrame) -> None:
        res = df_a * 2.0
        assert res._df["val"][0] == 20.0
        assert res.name == "(df_a * 2.0)"

    def test_div_scalar(self, df_a: DataFrame) -> None:
        res = df_a / 2.0
        assert res._df["val"][0] == 5.0
        assert res.name == "(df_a / 2.0)"

    def test_rsub_scalar(self, df_a: DataFrame) -> None:
        # Test 100 - df
        res = 100.0 - df_a
        assert res._df["val"][0] == 90.0  # 100 - 10
        assert res.name == "(100.0 - df_a)"

    def test_rdiv_scalar(self, df_a: DataFrame) -> None:
        # Test 100 / df
        res = 100.0 / df_a
        assert res._df["val"][0] == 10.0  # 100 / 10
        assert res.name == "(100.0 / df_a)"

    def test_complex_expression(self, df_a: DataFrame) -> None:
        # Test (df / df) - 1.0 (Common return calculation pattern)
        # Using self-operation aligns perfectly
        res_div = df_a / df_a  # Should be all 1.0
        res = res_div - 1.0  # Should be all 0.0

        assert res._df["val"][0] == 0.0
        assert res._df["val"][5] == 0.0
        assert res.name == "((df_a / df_a) - 1.0)"

    # --- Guards, dunders & misc ---

    def test_concat_non_dataframe_raises(self, df_a: DataFrame) -> None:
        with pytest.raises(TypeError, match="Can only concat with another DataFrame"):
            df_a.concat([1, 2, 3])  # type: ignore[arg-type]

    def test_align_non_dataframe_raises(self, df_a: DataFrame) -> None:
        with pytest.raises(TypeError, match="Can only align with another DataFrame"):
            df_a.align(123)  # type: ignore[arg-type]

    def test_subsecond_freq_raises(self) -> None:
        # Gaps below 1s round to 0 -> rejected (distinct from the duplicate case).
        df = pl.DataFrame(
            {
                "ts": [
                    datetime(2023, 1, 1, 0, 0, 0),
                    datetime(2023, 1, 1, 0, 0, 0, 500000),
                ],
                "val": [1.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="less than 1 second"):
            DataFrame(df, "ts")

    def test_str_includes_metadata(self, df_a: DataFrame) -> None:
        s = str(df_a)
        assert "DataFrame(name=df_a" in s
        assert "freq=2s" in s

    def test_getattr_non_callable_attr(self, df_a: DataFrame) -> None:
        # `shape` is a non-callable Polars attribute -> proxied through directly.
        assert df_a.shape == df_a._df.shape

    def test_radd_scalar(self, df_a: DataFrame) -> None:
        res = 5.0 + df_a
        assert res._df["val"][0] == df_a._df["val"][0] + 5.0
        assert res.name == "(5.0 + df_a)"

    def test_abs_and_dunder(self) -> None:
        df = DataFrame(
            pl.DataFrame(
                {
                    "ts": [
                        datetime(2023, 1, 1, 0, 0, 0),
                        datetime(2023, 1, 1, 0, 0, 1),
                    ],
                    "val": [-3.0, 4.0],
                }
            ),
            "ts",
            freq=1,
            name="d",
        )
        assert df.abs()._df["val"].to_list() == [3.0, 4.0]
        assert df.abs().name == "d.abs()"
        assert abs(df)._df["val"].to_list() == [3.0, 4.0]

    def test_aligned_op_non_dataframe_returns_notimplemented(
        self, df_a: DataFrame
    ) -> None:
        # The @auto_align wrapper short-circuits non-DataFrame operands.
        assert df_a._add_aligned(123) is NotImplemented
