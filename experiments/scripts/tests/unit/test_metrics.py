"""Unit tests for metrics module."""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / 'choice-learn'))

from utils.metrics import compute_rmse_with_freq, RMSECallback
from choice_learn.data import ChoiceDataset


class TestComputeRMSEWithFreq:
    """Test cases for compute_rmse_with_freq function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        n_samples = 10
        n_items = 3
        
        X = np.random.randint(0, 2, size=(n_samples, n_items))
        Y_freq = np.random.rand(n_samples, n_items)
        # Normalize to sum to 1
        Y_freq = Y_freq / Y_freq.sum(axis=1, keepdims=True)
        
        # Create dataset
        dataset = ChoiceDataset(
            items_features_by_choice=X[:, :, np.newaxis],
            available_items_by_choice=X.astype(int),
            choices=np.zeros(n_samples, dtype=int),
            shared_features_by_choice=(),
            items_features_by_choice_names=["availability"],
        )
        
        return X, Y_freq, dataset
    
    def test_compute_rmse_perfect_predictions(self, mock_model, sample_data):
        """Test RMSE when predictions match frequencies exactly."""
        X, Y_freq, dataset = sample_data
        
        # Mock model returns frequencies as predictions
        mock_model.predict_probas = Mock(return_value=Y_freq)
        
        rmse = compute_rmse_with_freq(mock_model, Y_freq, dataset=dataset)
        
        assert np.isclose(rmse, 0.0, atol=1e-6)
    
    def test_compute_rmse_with_dataset(self, mock_model, sample_data):
        """Test RMSE computation using dataset parameter."""
        X, Y_freq, dataset = sample_data
        
        predictions = np.random.rand(*Y_freq.shape)
        mock_model.predict_probas = Mock(return_value=predictions)
        
        rmse = compute_rmse_with_freq(mock_model, Y_freq, dataset=dataset)
        
        # Manually compute expected RMSE
        expected_rmse = np.sqrt(np.mean((predictions - Y_freq) ** 2))
        
        assert np.isclose(rmse, expected_rmse)
    
    def test_compute_rmse_with_X(self, mock_model, sample_data):
        """Test RMSE computation using X parameter (fallback)."""
        X, Y_freq, dataset = sample_data
        
        predictions = np.random.rand(*Y_freq.shape)
        mock_model.predict_probas = Mock(return_value=predictions)
        
        rmse = compute_rmse_with_freq(mock_model, Y_freq, X=X)
        
        # Should still compute RMSE correctly
        expected_rmse = np.sqrt(np.mean((predictions - Y_freq) ** 2))
        
        assert np.isclose(rmse, expected_rmse)
    
    def test_compute_rmse_raises_without_data(self, mock_model):
        """Test that function raises error when neither dataset nor X provided."""
        Y_freq = np.array([[0.5, 0.5]])
        
        with pytest.raises(ValueError, match="Either dataset or X must be provided"):
            compute_rmse_with_freq(mock_model, Y_freq)
    
    def test_compute_rmse_positive(self, mock_model, sample_data):
        """Test that RMSE is always non-negative."""
        X, Y_freq, dataset = sample_data
        
        # Create predictions that differ from frequencies
        predictions = Y_freq + np.random.randn(*Y_freq.shape) * 0.1
        mock_model.predict_probas = Mock(return_value=predictions)
        
        rmse = compute_rmse_with_freq(mock_model, Y_freq, dataset=dataset)
        
        assert rmse >= 0
    
    def test_compute_rmse_returns_float(self, mock_model, sample_data):
        """Test that RMSE returns a float."""
        X, Y_freq, dataset = sample_data
        
        predictions = np.random.rand(*Y_freq.shape)
        mock_model.predict_probas = Mock(return_value=predictions)
        
        rmse = compute_rmse_with_freq(mock_model, Y_freq, dataset=dataset)
        
        assert isinstance(rmse, (float, np.floating))


class TestRMSECallback:
    """Test cases for RMSECallback class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model wrapper."""
        model = Mock()
        model.predict_probas = Mock(return_value=np.array([[0.5, 0.5], [0.3, 0.7]]))
        return model
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        X = np.array([[1, 1], [1, 0]])
        dataset = ChoiceDataset(
            items_features_by_choice=X[:, :, np.newaxis],
            available_items_by_choice=X.astype(int),
            choices=np.array([0, 1]),
            shared_features_by_choice=(),
            items_features_by_choice_names=["availability"],
        )
        return dataset
    
    def test_callback_initialization(self, mock_model, sample_dataset):
        """Test that callback initializes correctly."""
        Y_freq = np.array([[0.5, 0.5], [0.3, 0.7]])
        
        callback = RMSECallback(mock_model, sample_dataset, Y_freq, log_freq=10)
        
        assert callback.model_wrapper == mock_model
        assert callback.dataset == sample_dataset
        assert np.array_equal(callback.Y_freq, Y_freq)
        assert callback.log_freq == 10
        assert callback.rmse_history == []
    
    def test_callback_on_epoch_end(self, mock_model, sample_dataset):
        """Test callback behavior at epoch end."""
        Y_freq = np.array([[0.5, 0.5], [0.3, 0.7]])
        
        callback = RMSECallback(mock_model, sample_dataset, Y_freq, log_freq=10)
        
        logs = {}
        callback.on_epoch_end(epoch=0, logs=logs)
        
        # Should have computed RMSE
        assert len(callback.rmse_history) == 1
        assert 'rmse' in logs
        assert isinstance(callback.rmse_history[0], (float, np.floating))
    
    def test_callback_multiple_epochs(self, mock_model, sample_dataset):
        """Test callback accumulates RMSE over multiple epochs."""
        Y_freq = np.array([[0.5, 0.5], [0.3, 0.7]])
        
        callback = RMSECallback(mock_model, sample_dataset, Y_freq, log_freq=10)
        
        # Simulate 5 epochs
        for epoch in range(5):
            callback.on_epoch_end(epoch=epoch, logs={})
        
        assert len(callback.rmse_history) == 5
    
    def test_callback_log_freq(self, mock_model, sample_dataset, capsys):
        """Test that callback respects log_freq for printing."""
        Y_freq = np.array([[0.5, 0.5], [0.3, 0.7]])
        
        callback = RMSECallback(mock_model, sample_dataset, Y_freq, log_freq=2)
        
        # First epoch should print (epoch == 0)
        callback.on_epoch_end(epoch=0, logs={})
        captured = capsys.readouterr()
        assert "RMSE:" in captured.out
        
        # Second epoch should not print (epoch == 1)
        callback.on_epoch_end(epoch=1, logs={})
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Third epoch should print (epoch + 1) % 2 == 0
        callback.on_epoch_end(epoch=2, logs={})
        captured = capsys.readouterr()
        assert "RMSE:" in captured.out
    
    def test_callback_stores_rmse_in_logs(self, mock_model, sample_dataset):
        """Test that callback adds RMSE to logs dictionary."""
        Y_freq = np.array([[0.5, 0.5], [0.3, 0.7]])
        
        callback = RMSECallback(mock_model, sample_dataset, Y_freq)
        
        logs = {}
        callback.on_epoch_end(epoch=0, logs=logs)
        
        assert 'rmse' in logs
        assert logs['rmse'] == callback.rmse_history[0]
