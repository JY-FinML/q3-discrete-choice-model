"""Data processing utilities for DeepHalo experiments."""

import numpy as np
import pandas as pd
from choice_learn.data import ChoiceDataset


def calc_freq(X, Y):
    """
    Calculate empirical choice frequencies for each unique choice set.
    
    For each unique availability pattern (choice set), compute the average
    choice probabilities across all instances of that pattern.
    
    Parameters
    ----------
    X : np.ndarray
        Availability masks, shape (n_samples, n_items)
    Y : np.ndarray
        One-hot encoded choices, shape (n_samples, n_items)
        
    Returns
    -------
    np.ndarray
        Frequency vectors, same shape as Y
    """
    # Convert to float for computation
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Find unique choice sets and their inverse mapping
    unique_X, inverse_indices = np.unique(X, axis=0, return_inverse=True)
    
    # Initialize frequency array
    Y_freq = np.zeros_like(Y)
    
    # For each unique choice set, compute average choice probabilities
    for k in range(unique_X.shape[0]):
        mask = (inverse_indices == k)
        avg_y = np.mean(Y[mask], axis=0)
        Y_freq[mask] = avg_y
    
    return Y_freq


def load_dataset(data_path, n_items, name="data"):
    """
    Load and prepare a single dataset.
    
    Parameters
    ----------
    data_path : str
        Path to CSV file
    n_items : int
        Number of items in the choice set
    name : str, optional
        Name for logging purposes (e.g., "train", "test")
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Availability masks array
        - 'Y': One-hot encoded choices array
        - 'y': Choice indices array
        - 'dataset': ChoiceDataset object
        - 'Y_freq': Empirical choice frequencies
    """
    print(f"Loading {name} data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Extract features
    X = df[[f'X{i}' for i in range(n_items)]].values
    Y = df[[f'Y{i}' for i in range(n_items)]].values
    y = np.argmax(Y, axis=1)
    
    print(f"{name.capitalize()} samples: {X.shape[0]}")
    
    # Create ChoiceDataset
    dataset = ChoiceDataset(
        items_features_by_choice=X[:, :, np.newaxis],
        available_items_by_choice=X.astype(int),
        choices=y,
        shared_features_by_choice=(),
        items_features_by_choice_names=["availability"],
    )
    
    print(f"Number of items: {dataset.get_n_items()}")
    
    # Compute empirical choice frequencies
    print(f"Computing empirical choice frequencies for {name}...")
    Y_freq = calc_freq(X, Y)
    
    return {
        'X': X,
        'Y': Y,
        'y': y,
        'dataset': dataset,
        'Y_freq': Y_freq,
    }