"""Integration tests for the training pipeline."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import yaml
import json
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / 'choice-learn'))

from utils import load_dataset, compute_rmse_with_freq, RMSECallback
from choice_learn.models import DeepHaloFeatureless


class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def sample_data_file(self):
        """Create a sample training data file."""
        n_items = 5
        n_samples = 100
        
        # Create synthetic data with guaranteed availability
        data = {}
        availability = []
        for _ in range(n_samples):
            # Ensure at least one item is available in each choice set
            avail = np.random.rand(n_items) > 0.3
            if not np.any(avail):
                # Force at least one item to be available
                avail[np.random.randint(n_items)] = True
            availability.append(avail)
        
        availability = np.array(availability)
        
        # Store availability
        for i in range(n_items):
            data[f'X{i}'] = availability[:, i].astype(int)
        
        # Create one-hot encoded choices (only from available items)
        choices = []
        for i in range(n_samples):
            available_indices = np.where(availability[i])[0]
            choice = np.random.choice(available_indices)
            choices.append(choice)
        
        choices = np.array(choices)
        for i in range(n_items):
            data[f'Y{i}'] = (choices == i).astype(int)
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path, n_items
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_data_loading_pipeline(self, sample_data_file):
        """Test complete data loading pipeline."""
        data_path, n_items = sample_data_file
        
        # Load dataset
        data = load_dataset(data_path, n_items, name="train")
        
        # Verify all components are present
        assert 'X' in data
        assert 'Y' in data
        assert 'y' in data
        assert 'dataset' in data
        assert 'Y_freq' in data
        
        # Verify data consistency
        assert data['X'].shape[1] == n_items
        assert data['Y'].shape[1] == n_items
        assert len(data['y']) == len(data['X'])
        
        # Verify frequencies are valid probabilities
        assert np.all(data['Y_freq'] >= 0)
        assert np.all(data['Y_freq'] <= 1)
    
    def test_model_training_pipeline(self, sample_data_file):
        """Test complete model training pipeline with small dataset."""
        data_path, n_items = sample_data_file
        
        # Load data
        data = load_dataset(data_path, n_items, name="train")
        
        # Create callback first (model_wrapper=None, will be set after model creation)
        rmse_callback = RMSECallback(
            model_wrapper=None,
            dataset=data['dataset'],
            Y_freq=data['Y_freq'],
            log_freq=2
        )
        
        # Create small model with callback
        model = DeepHaloFeatureless(
            n_items=n_items,
            depth=3,
            hidden_dim=32,
            block_types=['qua', 'qua'],
            optimizer='Adam',
            epochs=5,  # Small number for testing
            batch_size=32,
            lr=0.001,
            loss_type='mse',
            callbacks=[rmse_callback]
        )
        
        # Set model reference in callback
        rmse_callback.model_wrapper = model
        
        # Train model
        history = model.fit(
            data['dataset'],
            verbose=0
        )
        
        # Verify training completed
        assert 'train_loss' in history
        assert len(history['train_loss']) == 5
        assert len(rmse_callback.rmse_history) == 5
        
        # Verify RMSE was computed
        train_rmse = compute_rmse_with_freq(
            model,
            data['Y_freq'],
            dataset=data['dataset']
        )
        
        assert isinstance(train_rmse, (float, np.floating))
        assert train_rmse >= 0
    
    def test_model_save_and_load_pipeline(self, sample_data_file, temp_output_dir):
        """Test model saving and loading."""
        data_path, n_items = sample_data_file
        
        # Load data and train model
        data = load_dataset(data_path, n_items, name="train")
        
        model = DeepHaloFeatureless(
            n_items=n_items,
            depth=3,
            hidden_dim=32,
            block_types=['qua', 'qua'],
            epochs=3,
            batch_size=32
        )
        
        model.fit(data['dataset'], verbose=0)
        
        # Save model
        model_path = os.path.join(temp_output_dir, "test_model")
        model.save_model(model_path)
        
        # Verify model files were created
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = DeepHaloFeatureless.load_model(model_path)
        
        # Verify loaded model can make predictions
        original_pred = model.predict_probas(data['dataset'])
        loaded_pred = loaded_model.predict_probas(data['dataset'])
        
        # Predictions should be identical
        assert np.allclose(original_pred, loaded_pred, rtol=1e-5)
    
    def test_results_saving_pipeline(self, sample_data_file, temp_output_dir):
        """Test complete results saving pipeline."""
        data_path, n_items = sample_data_file
        
        # Load and train
        data = load_dataset(data_path, n_items, name="train")
        
        # Create callback first
        rmse_callback = RMSECallback(
            model_wrapper=None,
            dataset=data['dataset'],
            Y_freq=data['Y_freq']
        )
        
        # Create model with callback
        model = DeepHaloFeatureless(
            n_items=n_items,
            depth=3,
            hidden_dim=32,
            block_types=['qua', 'qua'],
            epochs=3,
            batch_size=32,
            callbacks=[rmse_callback]
        )
        
        # Set model reference
        rmse_callback.model_wrapper = model
        
        history = model.fit(data['dataset'], verbose=0)
        
        # Compute metrics
        train_rmse = compute_rmse_with_freq(model, data['Y_freq'], dataset=data['dataset'])
        
        # Create results dictionary
        results = {
            "config": {
                "param_set": "test",
                "depth": 3,
                "hidden_dim": 32,
            },
            "training_history": {
                "mse": [float(x) for x in history.get("train_loss", [])],
                "rmse": rmse_callback.rmse_history
            },
            "train_rmse_freq": float(train_rmse)
        }
        
        # Save results
        result_path = os.path.join(temp_output_dir, "test_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Verify file was created and contains valid data
        assert os.path.exists(result_path)
        
        with open(result_path, "r") as f:
            loaded_results = json.load(f)
        
        assert "config" in loaded_results
        assert "training_history" in loaded_results
        assert "train_rmse_freq" in loaded_results
        assert len(loaded_results["training_history"]["mse"]) == 3
        assert len(loaded_results["training_history"]["rmse"]) == 3
    
    def test_multiple_configurations_pipeline(self, sample_data_file, temp_output_dir):
        """Test training multiple model configurations."""
        data_path, n_items = sample_data_file
        
        # Load data once
        data = load_dataset(data_path, n_items, name="train")
        
        # Train multiple configurations
        configs = [
            {'depth': 2, 'hidden_dim': 16},
            {'depth': 3, 'hidden_dim': 32},
        ]
        
        all_results = []
        
        for config in configs:
            # Create callback
            rmse_callback = RMSECallback(
                model_wrapper=None,
                dataset=data['dataset'],
                Y_freq=data['Y_freq']
            )
            
            # Create model with callback
            model = DeepHaloFeatureless(
                n_items=n_items,
                depth=config['depth'],
                hidden_dim=config['hidden_dim'],
                block_types=['qua'] * (config['depth'] - 1),
                epochs=2,
                batch_size=32,
                callbacks=[rmse_callback]
            )
            
            # Set model reference
            rmse_callback.model_wrapper = model
            
            history = model.fit(data['dataset'], verbose=0)
            
            train_rmse = compute_rmse_with_freq(model, data['Y_freq'], dataset=data['dataset'])
            
            results = {
                "config": {
                    "param_set": "test",
                    **config
                },
                "training_history": {
                    "mse": [float(x) for x in history.get("train_loss", [])],
                    "rmse": rmse_callback.rmse_history
                },
                "train_rmse_freq": float(train_rmse)
            }
            
            all_results.append(results)
        
        # Verify all configurations trained successfully
        assert len(all_results) == 2
        for result in all_results:
            assert len(result["training_history"]["mse"]) == 2
            assert len(result["training_history"]["rmse"]) == 2
            assert result["train_rmse_freq"] >= 0
    