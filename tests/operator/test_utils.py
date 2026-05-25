import numpy as np

from ctalearn.operator._utils import numba_rolling_std


def test_rolling_std_basic() -> None:
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    got = numba_rolling_std(arr, window=5, ddof=1)
    assert np.isclose(got[-1], np.std([0, 1, 2, 3, 4], ddof=1))


def test_rolling_std_large_offset_precision() -> None:
    # Catastrophic cancellation: large offset, tiny variance.
    # Naive E[x^2]-E[x]^2 loses all precision (collapses to ~0 / wrong value).
    exp = np.std([0, 1, 2, 3, 4], ddof=1)
    for offset in (1e8, 1e10, 1e12):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0]) + offset
        got = numba_rolling_std(arr, window=5, ddof=1)[-1]
        assert np.isclose(got, exp, rtol=1e-6), f"offset={offset}: {got} != {exp}"


def test_rolling_std_shorter_than_window() -> None:
    # n < window -> all NaN, no computation.
    got = numba_rolling_std(np.array([1.0, 2.0]), window=5, ddof=1)
    assert got.shape == (2,)
    assert np.isnan(got).all()


def test_rolling_std_sliding_window() -> None:
    # window < n -> values slide out, exercising the eviction branch.
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    window = 3
    got = numba_rolling_std(arr, window=window, ddof=1)
    assert np.isnan(got[: window - 1]).all()  # warm-up
    exp = [
        np.std(arr[i - window + 1 : i + 1], ddof=1) for i in range(window - 1, len(arr))
    ]
    np.testing.assert_allclose(got[window - 1 :], exp)


def test_rolling_std_nan_in_window() -> None:
    # Any window containing a NaN -> NaN; clean windows compute normally.
    arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0])
    got = numba_rolling_std(arr, window=3, ddof=1)
    assert np.isclose(got[2], np.std([1, 2, 3], ddof=1))  # clean
    assert np.isnan(got[3])  # NaN enters window
    assert np.isnan(got[4]) and np.isnan(got[5])  # NaN still in window
    assert np.isclose(got[6], np.std([5, 6, 7], ddof=1))  # NaN slid out
