"""Metrics and evaluation utilities for DeepHalo experiments."""

import numpy as np
import tensorflow as tf
from choice_learn.data import ChoiceDataset


def compute_rmse_with_freq(trained_model, Y_freq, dataset=None, X=None):
    """
    Compute RMSE between model predictions and empirical choice frequencies.
    
    This follows the original DeepHalo paper methodology where RMSE is computed
    against empirical frequencies rather than individual choices.
    
    Parameters
    ----------
    trained_model : DeepHaloFeatureless
        Trained model
    Y_freq : np.ndarray
        Empirical choice frequency vectors
    dataset : ChoiceDataset, optional
        Pre-created ChoiceDataset (preferred for efficiency)
    X : np.ndarray, optional
        Availability masks (only if dataset not provided)
        
    Returns
    -------
    float
        RMSE value
    """
    # Get predictions using pre-created dataset or create temporary one
    if dataset is not None:
        predictions = trained_model.predict_probas(dataset)
    elif X is not None:
        # Fallback: create temporary dataset
        X_reshaped = X[:, :, np.newaxis]
        temp_dataset = ChoiceDataset(
            items_features_by_choice=X_reshaped,
            available_items_by_choice=X.astype(int),
            choices=np.zeros(len(X), dtype=int),  # Dummy choices
            shared_features_by_choice=(),
            items_features_by_choice_names=["availability"],
        )
        predictions = trained_model.predict_probas(temp_dataset)
    else:
        raise ValueError("Either dataset or X must be provided")
    
    # Compute MSE then take square root
    mse = np.mean((predictions - Y_freq) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


class RMSECallback(tf.keras.callbacks.Callback):
    """Custom callback to compute RMSE during training."""
    
    def __init__(self, model_wrapper, dataset, Y_freq, log_freq=50):
        """
        Parameters
        ----------
        model_wrapper : DeepHaloFeatureless
            The model wrapper object
        dataset : ChoiceDataset
            Pre-created training ChoiceDataset
        Y_freq : np.ndarray
            Empirical frequency vectors
        log_freq : int
            Print RMSE every log_freq epochs (always computes every epoch)
        """
        super().__init__()
        self.model_wrapper = model_wrapper
        self.dataset = dataset
        self.Y_freq = Y_freq
        self.log_freq = log_freq
        self.rmse_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        """Compute RMSE at the end of each epoch."""
        rmse = compute_rmse_with_freq(
            self.model_wrapper, 
            self.Y_freq, 
            dataset=self.dataset
        )
        # Convert to Python float for JSON serialization
        self.rmse_history.append(float(rmse))
        
        # Add RMSE to logs dict if provided
        if logs is not None:
            logs['rmse'] = float(rmse)
        
        # Print every log_freq epochs
        if epoch % self.log_freq == 0:
            print(f" - RMSE: {rmse:.6f}", end='')
