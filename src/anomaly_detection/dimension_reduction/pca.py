import numpy as np

def dynamic_augmentation(X, s):
    """
    Build dynamic augmented matrix:
    [x(t), x(t-1), ..., x(t-s)]
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, m_features)
        Original time series matrix. Row t = x(t).
    s : int
        Lag time.
    
    Returns
    -------
    Xs : ndarray of shape (n_samples - s, m_features * (s + 1))
    """
    X = np.asarray(X)
    n, m = X.shape

    if s < 0:
        raise ValueError("s must be >= 0")
    if s >= n:
        raise ValueError("s must be smaller than number of samples")

    blocks = []
    for lag in range(s + 1):
        blocks.append(X[s - lag : n - lag, :])

    Xs = np.hstack(blocks)
    return Xs

def inverse_dynamic_augmentation(Xs, s, m):
    """
    Reconstruct original time series X from augmented matrix Xs.

    Parameters
    ----------
    Xs : ndarray (n-s, m*(s+1))
        Augmented matrix
    s : int
        Lag used in augmentation
    m : int
        Number of original features

    Returns
    -------
    X : ndarray (n, m)
        Reconstructed original matrix
    """

    n_minus_s = Xs.shape[0]
    n = n_minus_s + s

    X = np.zeros((n, m))
    counts = np.zeros(n)

    for lag in range(s + 1):
        block = Xs[:, lag*m:(lag+1)*m]
        start = s - lag
        end = n - lag

        X[start:end] += block
        counts[start:end] += 1

    X /= counts[:, None]

    return X

class lag_augmented:
    """
    Class to implement the dynamic augmentation matrix
    parameters :
        lag (int) : the lag used for lag-augmented matrix
        m (int) : the number of variables of the dataset
    """
    def __init__(self, lag, m):
        self.lag = lag
        self.m = m
    
    def transform(self, X):
        return dynamic_augmentation(X, self.lag)
    
    def inverse_transform(self, X):
        return inverse_dynamic_augmentation(X, self.lag, self.m)