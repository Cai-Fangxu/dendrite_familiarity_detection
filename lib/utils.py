import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def capacity_numba(transient_record: np.ndarray, steady_record: np.ndarray) -> np.ndarray:
    """
    transient_record : (n_steps1, n_bs, n_patterns)  float64
    steady_record : (n_steps3, n_bs, n_patterns)  float64
    returns
    --------
    capacity : (n_bs,) float64
    """
    n_steps1, n_bs, n_patterns = transient_record.shape
    k01 = max(0, int(np.ceil(0.01 * n_patterns)) - 1)        # 1-st-perc index

    capacity = np.zeros(n_bs)
    eps      = 1e-5

    for b_idx in prange(n_bs):                                   # parallel over batches
        # --- 99-th percentile on the steady slice
        steady   = steady_record[:, b_idx, :].ravel()
        k99      = max(0, int(np.ceil(0.99 * steady.size)) - 1)
        upper99  = np.partition(steady, k99)[k99] + eps

        # --- stream through time; no big temp arrays needed
        sum_crossings = 0.0
        cnt_crossings = 0

        # initialise diff_(t=0)
        prev = np.partition(transient_record[0, b_idx, :], k01)[k01] - upper99

        for t in range(1, n_steps1):
            curr = np.partition(transient_record[t, b_idx, :], k01)[k01] - upper99
            if prev >= 0.0 and curr < 0.0:                   # zero-crossing test
                sum_crossings += t - 1                       # position at prev step
                cnt_crossings += 1
            prev = curr

        if cnt_crossings:
            capacity[b_idx] = sum_crossings / cnt_crossings      # average position

    return capacity