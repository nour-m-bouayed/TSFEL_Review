from tsfel.feature_extraction.features_utils import set_domain
import pandas as pd
import numpy as np
from math import factorial, log
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch
from numba import njit
import inspect



def embed_signal(x, order=3, delay=1):
    x = np.asarray(x)
    N = x.shape[-1]
    if x.ndim == 1:
        # 1D array (n_times)
        Y = np.lib.stride_tricks.as_strided(x, shape=(N - (order - 1) * delay, order), strides=(x.itemsize * delay, x.itemsize))
        return Y
    else:
        # 2D array (signal_indice, n_times)
        embed_signal_length = N - (order - 1) * delay
        indice = np.array([[i * delay, i * delay + embed_signal_length] for i in range(order)])
        Y = np.lib.stride_tricks.as_strided(x, shape=(x.shape[0], embed_signal_length, order), strides=(x.strides[1], x.strides[1] * delay, x.strides[0]))
        return Y
    
# @set_domain("domain", "spectral")
# def spectral_entropy(signal, fs, nperseg=None):
#     """Spectral Entropy."""
#     x = np.asarray(signal)
#     _, psd = welch(x, fs, nperseg=nperseg, axis=-1)
#     psd_norm = psd / psd.sum(axis=-1, keepdims=True)
#     spect_en = -np.sum(psd_norm * np.log2(psd_norm), axis=-1)
#     spect_en /= np.log2(psd_norm.shape[-1]) #normalise
#     return float(spect_en)

@set_domain("domain", "information_t")
def permutation_entropy(signal, order=3, delay=1):
    x = np.array(signal)
    
    # Generate hash multiplier
    hashmult = np.power(order, range(order))
    
    #Create sorted indices for embedded signal
    sorted_idx = np.argsort(embed_signal(x, order=order, delay=delay))
    
    #Associate unique integer to each permutation
    hashval = np.sum(sorted_idx * hashmult, axis=1)
    
    # Count unique permutations
    _, counts = np.unique(hashval, return_counts=True)
    
    #Calculate probability distribution
    probabilities = counts / counts.sum()
    
    # Calculate permutation entropy
    pe = -np.sum(probabilities * np.log2(probabilities))
    pe /= np.log2(factorial(order))
    return pe

@set_domain("domain", "information_t")
def approximate_entropy(signal, order=2, metric="chebyshev"):
    # r set to 0.2 times the standard deviation of the signal
    x = np.asarray(signal)
    r = 0.2 * np.std(x, ddof=0)
    
    # Compute phi(order, r) and phi(order + 1, r)
    phi = np.zeros(2)
    
    # Compute phi(order, r)
    emb_data1 = embed_signal(x, order, 1)
    count1 = (
        KDTree(emb_data1, metric=metric)
        .query_radius(emb_data1, r, count_only=True)
        .astype(np.float64)
    )
    
    # Compute phi(order + 1, r)
    emb_data2 = embed_signal(x, order + 1, 1)
    count2 = (
        KDTree(emb_data2, metric=metric)
        .query_radius(emb_data2, r, count_only=True)
        .astype(np.float64)
    )

    phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
    phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))

    return np.subtract(phi[0], phi[1])


@set_domain("domain", "information_t")
def sample_entropy(signal, order=2):
    """Sample Entropy."""
    r = 0.2 * np.std(signal)  # Avoid using ddof in Numba
    x = np.asarray(signal, dtype=np.float64)
    size = x.size

    numerator = 0
    denominator = 0

    for offset in range(1, size - order):
        n_numerator = int(np.abs(x[order] - x[order + offset]) >= r)
        n_denominator = 0

        for idx in range(order):
            n_numerator += np.abs(x[idx] - x[idx + offset]) >= r
            n_denominator += np.abs(x[idx] - x[idx + offset]) >= r

        if n_numerator == 0:
            numerator += 1
        if n_denominator == 0:
            denominator += 1

        prev_in_diff = int(np.abs(x[order] - x[offset + order]) >= r)
        for idx in range(1, size - offset - order):
            out_diff = int(np.abs(x[idx - 1] - x[idx + offset - 1]) >= r)
            in_diff = int(np.abs(x[idx + order] - x[idx + offset + order]) >= r)
            n_numerator += in_diff - out_diff
            n_denominator += prev_in_diff - out_diff
            prev_in_diff = in_diff

            if n_numerator == 0:
                numerator += 1
            if n_denominator == 0:
                denominator += 1

    if denominator == 0:
        return 0  # use 0/0 == 0
    elif numerator == 0:
        return np.inf
    else:
        return -log(numerator / denominator)

@set_domain("domain", "temporal")
def is_it_weekend(signal, parameters=None):
    """
    Returns 0 if the span of the time series is a weekday, 1 if it's a weekend day.

    Parameters
    ----------
    signal : pandas.Series
        The time series to calculate the feature of. Index must be a datetime type.

    Returns
    -------
    int
        0 for weekdays, 1 for weekends.
    """

    if isinstance(signal, pd.Series):
        # Ensure the index is a pandas datetime type
        if not isinstance(signal, pd.DatetimeIndex):
            raise ValueError("Index of the signal must be a pandas DatetimeIndex.")

        # Calculate the total time span of the time series
        total_span = signal.index[-1] - signal.index[0]

        # If the total span is a day, check if it's a weekend or a weekday
        if pd.Timedelta('0.8 days')< total_span < pd.Timedelta('1.2 days'):
            # Get the day of the week (Monday=0, Sunday=6)
            day_of_week = signal.index[0].dayofweek
        
            # Return 1 if it's a weekend (Saturday or Sunday), otherwise 0
            return 1 if day_of_week >= 5 else 0
        else:
            raise ValueError("The total span of the time series is not within a single day.")
    else:
    
        return 0

