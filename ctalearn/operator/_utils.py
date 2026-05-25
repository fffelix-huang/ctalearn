import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True, nogil=True, parallel=False)  # type: ignore[untyped-decorator]
def numba_rolling_std(
    arr: NDArray[np.float64], window: int, ddof: int = 1
) -> NDArray[np.float64]:
    """
    Rolling std in O(N).

    Accumulates sums of (x - K) for a constant offset K (the first finite
    value). Variance is invariant to this shift, but keeping the magnitudes
    small avoids the catastrophic cancellation that E[x^2] - E[x]^2 suffers
    when |x| is large relative to its variance.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return out

    # Constant offset to center the data (first finite value).
    k = 0.0
    for i in range(n):
        if not np.isnan(arr[i]):
            k = arr[i]
            break

    sum_x, sum_xx = 0.0, 0.0

    nan_count = window

    for i in range(n):
        if i < window or np.isnan(arr[i - window]):
            nan_count -= 1
        else:
            d_out = arr[i - window] - k
            sum_x -= d_out
            sum_xx -= d_out * d_out

        if np.isnan(arr[i]):
            nan_count += 1
        else:
            d_in = arr[i] - k
            sum_x += d_in
            sum_xx += d_in * d_in

        if nan_count > 0:
            continue

        mean = sum_x / window
        var = (sum_xx - sum_x * mean) / (window - ddof)
        out[i] = np.sqrt(max(var, 0.0))

    return out
