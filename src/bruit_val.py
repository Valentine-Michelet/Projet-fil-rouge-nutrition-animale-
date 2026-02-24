import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_gaussian_noise(X, noise_std=0.02, random_state=None):
    """
    Add Gaussian noise to feature matrix X only.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    noise_std : float
        Standard deviation of Gaussian noise
    random_state : int or None
        Seed for reproducibility

    Returns
    -------
    X_noisy : same type as X
    """
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0, scale=noise_std, size=X.shape)

    if isinstance(X, pd.DataFrame):
        X_noisy = X.copy()
        X_noisy += noise
        return X_noisy
    else:
        return X + noise


def add_noise_after_split(
    X_train,
    X_test,
    noise_std=0.02,
    random_state=42,
    noise_on="train"  # "train", "test", "both"
):
    """
    Apply Gaussian noise AFTER train/test split.
    Targets are not affected.
    """

    if noise_on in ["train", "both"]:
        X_train = add_gaussian_noise(
            X_train,
            noise_std=noise_std,
            random_state=random_state
        )

    if noise_on in ["test", "both"]:
        X_test = add_gaussian_noise(
            X_test,
            noise_std=noise_std,
            random_state=random_state + 1
        )

    return X_train, X_test


