"""Unit tests for data_utils module."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.data_utils import calc_freq, load_dataset


class TestCalcFreq:
    """Test cases for calc_freq function."""
    
    def test_calc_freq_simple_case(self):
        """Test calc_freq with a simple case."""
        # Create sample data: 2 items, 4 samples
        X = np.array([
            [1, 1],  # Choice set 1
            [1, 1],  # Choice set 1 (duplicate)
            [1, 0],  # Choice set 2
            [1, 0],  # Choice set 2 (duplicate)
        ])
        
        Y = np.array([
            [1, 0],  # Chose item 0
            [0, 1],  # Chose item 1
            [1, 0],  # Chose item 0
            [1, 0],  # Chose item 0
        ])
        
        Y_freq = calc_freq(X, Y)
        
        # For choice set 1 [1,1]: average of [1,0] and [0,1] = [0.5, 0.5]
        assert np.allclose(Y_freq[0], [0.5, 0.5])
        assert np.allclose(Y_freq[1], [0.5, 0.5])
        
        # For choice set 2 [1,0]: average of [1,0] and [1,0] = [1.0, 0.0]
        assert np.allclose(Y_freq[2], [1.0, 0.0])
        assert np.allclose(Y_freq[3], [1.0, 0.0])
    
    def test_calc_freq_all_unique(self):
        """Test calc_freq when all choice sets are unique."""
        X = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ])
        
        Y = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        
        Y_freq = calc_freq(X, Y)
        
        # Each choice set is unique, so frequencies should match original choices
        assert np.allclose(Y_freq, Y)
    
    def test_calc_freq_all_same(self):
        """Test calc_freq when all choice sets are identical."""
        X = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
        ])
        
        Y = np.array([
            [1, 0],
            [0, 1],
            [1, 0],
        ])
        
        Y_freq = calc_freq(X, Y)
        
        # All should have the same frequency: mean of Y
        expected = np.array([2/3, 1/3])
        for i in range(3):
            assert np.allclose(Y_freq[i], expected)
    
    def test_calc_freq_output_shape(self):
        """Test that output shape matches input shape."""
        X = np.random.randint(0, 2, size=(100, 5))
        Y = np.random.randint(0, 2, size=(100, 5))
        
        Y_freq = calc_freq(X, Y)
        
        assert Y_freq.shape == Y.shape
    
    def test_calc_freq_dtype(self):
        """Test that output dtype is float32."""
        X = np.array([[1, 0], [1, 0]])
        Y = np.array([[1, 0], [0, 1]])
        
        Y_freq = calc_freq(X, Y)
        
        assert Y_freq.dtype == np.float32


class TestLoadDataset:
    """Test cases for load_dataset function."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a temporary CSV file for testing."""
        n_items = 3
        n_samples = 10
        
        # Create sample data
        data = {}
        for i in range(n_items):
            data[f'X{i}'] = np.random.randint(0, 2, n_samples)
        
        # Create one-hot encoded choices
        choices = np.random.randint(0, n_items, n_samples)
        for i in range(n_items):
            data[f'Y{i}'] = (choices == i).astype(int)
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path, n_items, n_samples
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_load_dataset_returns_dict(self, sample_csv):
        """Test that load_dataset returns a dictionary."""
        csv_path, n_items, _ = sample_csv
        
        result = load_dataset(csv_path, n_items, name="test")
        
        assert isinstance(result, dict)
    
    def test_load_dataset_required_keys(self, sample_csv):
        """Test that result contains all required keys."""
        csv_path, n_items, _ = sample_csv
        
        result = load_dataset(csv_path, n_items, name="test")
        
        required_keys = ['X', 'Y', 'y', 'dataset', 'Y_freq']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_load_dataset_array_shapes(self, sample_csv):
        """Test that arrays have correct shapes."""
        csv_path, n_items, n_samples = sample_csv
        
        result = load_dataset(csv_path, n_items, name="test")
        
        assert result['X'].shape == (n_samples, n_items)
        assert result['Y'].shape == (n_samples, n_items)
        assert result['y'].shape == (n_samples,)
        assert result['Y_freq'].shape == (n_samples, n_items)
    
    def test_load_dataset_choice_consistency(self, sample_csv):
        """Test that choice indices match one-hot encoding."""
        csv_path, n_items, n_samples = sample_csv
        
        result = load_dataset(csv_path, n_items, name="test")
        
        # y should match argmax of Y
        assert np.array_equal(result['y'], np.argmax(result['Y'], axis=1))
    
    def test_load_dataset_choice_dataset_type(self, sample_csv):
        """Test that dataset is a ChoiceDataset instance."""
        csv_path, n_items, _ = sample_csv
        
        result = load_dataset(csv_path, n_items, name="test")
        
        from choice_learn.data import ChoiceDataset
        assert isinstance(result['dataset'], ChoiceDataset)
    
    def test_load_dataset_freq_probabilities(self, sample_csv):
        """Test that frequencies sum to approximately 1 for each sample."""
        csv_path, n_items, _ = sample_csv
        
        result = load_dataset(csv_path, n_items, name="test")
        
        # Frequencies should sum to 1 (or close to it) for each sample
        freq_sums = result['Y_freq'].sum(axis=1)
        assert np.allclose(freq_sums, 1.0, atol=1e-5)
