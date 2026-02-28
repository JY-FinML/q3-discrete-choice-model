import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from synthetic_data_generation import (
    generate_probability_list,
    generate_one_hot_batch,
    generate_data,
    save_dataset_to_csv,
)

def test_generate_probability_list_all_zeros():
    '''Test that generate_probability_list returns all zeros when input is all zeros.'''
    result = generate_probability_list([0, 0, 0])
    assert result == [0.0, 0.0, 0.0]

def test_generate_probability_list_some_ones():
    '''Test that generate_probability_list returns correct probabilities for a subset with some ones.'''
    np.random.seed(0)
    result = generate_probability_list([1, 0, 1])
    assert result[1] == 0.0
    assert abs(sum(result) - 1.0) < 1e-8
    assert result[0] > 0 and result[2] > 0

def test_generate_one_hot_batch_shape_and_sum():
    '''Test that generate_one_hot_batch returns a batch with correct shape and one-hot encoding.'''
    np.random.seed(0)
    probs = [0.2, 0.3, 0.5]
    batch = generate_one_hot_batch(probs, 10)
    assert batch.shape == (10, 3)
    assert np.all(batch.sum(axis=1) == 1)

def test_generate_data_shapes():
    '''Test that generate_data returns arrays with correct shapes and properties.'''
    np.random.seed(0)
    offer_set = [0, 1, 2]
    max_size = 2
    min_size = 2
    num_samples = 5
    X, Y = generate_data(offer_set, max_size, min_size, num_samples)
    # There are 3 choose 2 = 3 subsets, each repeated 5 times
    assert X.shape == (15, 3)
    assert Y.shape == (15, 3)
    # Each row in X should be a binary vector
    assert np.all((X == 0) | (X == 1))
    # Each row in Y should be one-hot
    assert np.all(Y.sum(axis=1) == 1)

def test_generate_data_and_save(tmp_path):
    '''Integration test for generate_data and save_dataset_to_csv functions.'''
    np.random.seed(0)
    offer_set = [0, 1, 2]
    X, Y = generate_data(offer_set, 2, 2, 2)
    file_path = tmp_path / "test.csv"
    save_dataset_to_csv(X, Y, offer_set, str(file_path))
    import pandas as pd
    df = pd.read_csv(file_path)
    assert df.shape == (6, 6)  # 3 subsets * 2 samples, 3 X + 3 Y columns
    assert all(col in df.columns for col in ["X0", "X1", "X2", "Y0", "Y1", "Y2"])