"""
Training script for DeepHalo Featureless model.

This script trains multiple configurations of the DeepHaloFeatureless model
with different depths and hidden dimensions, evaluates them, and generates
summary plots.
"""

import os
import sys
import json
import yaml
import numpy as np
import tensorflow as tf

# Add choice-learn to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'choice-learn'))

from choice_learn.models import DeepHaloFeatureless
from utils import (
    load_dataset,
    compute_rmse_with_freq,
    RMSECallback,
    plot_training_summary
)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def train_model_config(model_config, train_config, data, verbose=1):
    """
    Train a single model configuration.
    
    Parameters
    ----------
    model_config : dict
        Model configuration (depth, hidden_dim, block_types)
    train_config : dict
        Training configuration (optimizer, epochs, batch_size, lr, loss_type)
    data : dict
        Data dictionary from load_and_prepare_data
    verbose : int
        Verbosity level for training
        
    Returns
    -------
    dict
        Results dictionary with config, history, and metrics
    """
    depth = model_config['depth']
    hidden_dim = model_config['hidden_dim']
    param_set = model_config['param_set']
    
    print(f"\n[{param_set}] Training: depth={depth}, hidden_dim={hidden_dim}")
    
    # Create RMSE callback (needs to be created before model to pass to __init__)
    # We'll set the model reference after model creation
    rmse_callback = RMSECallback(
        model_wrapper=None,  # Will be set after model creation
        dataset=data['dataset'],
        Y_freq=data['Y_freq'], 
        log_freq=50
    )
    
    # Create model with callback
    model = DeepHaloFeatureless(
        depth=depth,
        hidden_dim=hidden_dim,
        block_types=model_config['block_types'],
        optimizer=train_config['optimizer'],
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        lr=train_config['lr'],
        loss_type=train_config['loss_type'],
        callbacks=[rmse_callback]
    )
    
    # Set model reference in callback
    rmse_callback.model_wrapper = model
    
    # Train model
    print("Training...")
    history = model.fit(
        data['dataset'], 
        verbose=verbose
    )
    
    # Evaluate on training set
    print("\nEvaluating...")
    train_rmse = compute_rmse_with_freq(model, data['Y_freq'], dataset=data['dataset'])
    
    print(f"Train RMSE: {train_rmse:.6f}")
    
    # Prepare results
    results = {
        "config": {
            "param_set": param_set,
            "depth": depth,
            "hidden_dim": hidden_dim,
            **train_config
        },
        "training_history": {
            "mse": [float(x) for x in history.get("train_loss", [])],
            "rmse": rmse_callback.rmse_history
        },
        "train_rmse_freq": float(train_rmse)
    }
    
    return model, results


def save_results(results, model, param_set, depth, hidden_dim, output_dir, save_model=True):
    """
    Save training results and model.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    model : DeepHaloFeatureless
        Trained model
    param_set : str
        Parameter set name
    depth : int
        Model depth
    hidden_dim : int
        Hidden dimension
    output_dir : str
        Output directory
    save_model : bool
        Whether to save the model
    """
    # Save results JSON
    result_filename = f"{param_set}_depth{depth}_dim{hidden_dim}_results.json"
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {result_path}")
    
    # Save model
    if save_model:
        model_filename = f"{param_set}_depth{depth}_dim{hidden_dim}_model"
        model_path = os.path.join(output_dir, model_filename)
        model.save_model(model_path)
        print(f"Model saved to: {model_path}")


def main():
    """Main training function."""
    # Load configuration
    config_path = "experiments/configs/deephalo_featureless.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = config["output"]["result_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    data = load_dataset(
        data_path=config["data"]["train_path"],
        n_items=config["data"]["n_items"],
        name="train"
    )
    
    # Training configuration
    train_config = {
        "optimizer": config["training"]["optimizer"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "lr": config["training"]["lr"],
        "loss_type": config["training"]["loss_type"]
    }
    
    # Train all configurations
    all_results = []
    param_sets = [
        ("param_200k", config["parameters"]["param_200k"]),
        ("param_500k", config["parameters"]["param_500k"])
    ]
    
    for param_set_name, param_list in param_sets:
        print(f"\n{'='*60}")
        print(f"Training {param_set_name} configurations")
        print(f"{'='*60}")
        
        for idx, params in enumerate(param_list, 1):
            depth = params["depth"]
            hidden_dim = params["hidden_dim"]
            
            # Prepare model config
            model_config = {
                "param_set": param_set_name,
                "depth": depth,
                "hidden_dim": hidden_dim,
                "block_types": [config["model"]["block_types"]] * (depth - 1)
            }
            
            # Train model
            model, results = train_model_config(model_config, train_config, data)
            
            # Save results
            save_results(
                results, model, param_set_name, depth, hidden_dim, 
                output_dir, save_model=config["output"]["save_model"]
            )
            
            # Store for plotting
            all_results.append(results)
    
    # Generate summary
    print(f"\n{'='*60}")
    print("All training completed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")
    
    # Create summary plots
    if config["output"].get("save_plots", True):
        plot_training_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
