import numpy as np


def min_max_normalization(X: np.ndarray):
    range_value = np.max(X, axis=0) - np.min(X, axis=0)
    print("range_value.shape: ", range_value.shape)
    normalized_X = (X - np.min(X, axis=0))/range_value
    print("normalized_X.shape:", normalized_X.shape)
    return normalized_X