"""DeepHalo Feature-Based Model

This module implements "Deep Context-Dependent Choice Model", a feature-based ResNet-like deep neural network model for discrete choice prediction.
The model uses residual blocks with nonlinear transformations to learn complex choice patterns (including halo effects)
from item features. Unlike the featureless version, this model processes actual feature vectors for each item.

Architecture:
- Basic encoder: 3-layer MLP that encodes item features
- L residual blocks: Each block performs:
  1. Global aggregation: Linear projection → average pooling across items → reshape for broadcasting
  2. Nonlinear transformation: Multi-head transformation (H heads) on encoded features
  3. Element-wise multiplication and aggregation: Multiply with pooled features → sum over heads
  4. Residual connection: Add back to input
- Final linear layer: Projects to logits
- Softmax with availability masking: Masks unavailable items with -inf before softmax

The model is based on the PyTorch implementation and adapted for TensorFlow and choice-learn library.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .base_model import ChoiceModel


class NonlinearTransformation(tf.keras.layers.Layer):
    """Nonlinear Transformation Layer.
    
    Implements a multi-head nonlinear transformation:
    - Projects to H heads
    - Applies ReLU activation
    - Projects back to original dimension
    - Applies layer normalization
    
    Parameters
    ----------
    H : int
        Number of heads for the transformation
    embed_dim : int
        Embedding dimension
    dropout : float, optional
        Dropout rate, by default 0.0
    """
    
    def __init__(self, H, embed_dim, dropout=0.0, **kwargs):
        """Initialize NonlinearTransformation layer.
        
        Parameters
        ----------
        H : int
            Number of heads
        embed_dim : int
            Embedding dimension
        dropout : float, optional
            Dropout rate, by default 0.0
        """
        super().__init__(**kwargs)
        self.H = H
        self.embed_dim = embed_dim
        self.fc1 = tf.keras.layers.Dense(embed_dim * H, use_bias=True)
        self.fc2 = tf.keras.layers.Dense(embed_dim, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, X, training=False):
        """Forward pass.
        
        Parameters
        ----------
        X : tf.Tensor
            Input tensor of shape (batch_size, n_items, embed_dim)
        training : bool, optional
            Whether in training mode (for dropout), by default False
            
        Returns
        -------
        tf.Tensor
            Transformed tensor of shape (batch_size, n_items, H, embed_dim)
        """
        B = tf.shape(X)[0]
        n = tf.shape(X)[1]
        
        # Project to H heads: (B, n, embed*H)
        out = self.fc1(X)
        # Reshape to (B, n, H, embed)
        out = tf.reshape(out, [B, n, self.H, self.embed_dim])
        # Apply ReLU
        out = tf.nn.relu(out)
        # Apply dropout
        out = self.dropout(out, training=training)
        # Project back: (B, n, H, embed) -> (B, n, H, embed)
        # Note: fc2 operates on last dimension
        out = self.fc2(out)
        # Layer normalization on last dimension
        out = self.layer_norm(out)
        
        return out


class DeepHaloFeature(ChoiceModel):
    """DeepHalo Feature-Based Model.
    
    A deep neural network for discrete choice modeling that uses item features
    to learn complex choice patterns with halo effects through residual transformations.
    
    Parameters
    ----------
    n_items : int, optional
        Number of items in the choice set. If not provided, will be inferred from data.
    input_dim : int, optional
        Dimension of input features per item. If not provided, will be inferred from data.
    H : int, optional
        Number of heads in nonlinear transformation, by default 4
    L : int, optional
        Number of residual layers, by default 3
    embed_dim : int, optional
        Embedding dimension, by default 128
    dropout : float, optional
        Dropout rate, by default 0.0
    optimizer : str, optional
        Optimizer to use ('Adam', 'SGD', 'lbfgs', etc.), by default 'Adam'
    lr : float, optional
        Learning rate, by default 0.001
    epochs : int, optional
        Number of training epochs, by default 500
    batch_size : int, optional
        Batch size for training, by default 1024
    **kwargs
        Additional arguments passed to ChoiceModel base class
        
    Example
    -------
    >>> # With explicit dimensions
    >>> model = DeepHaloFeature(
    ...     n_items=20,
    ...     input_dim=10,
    ...     H=4,
    ...     L=3,
    ...     embed_dim=128,
    ...     optimizer='Adam',
    ...     lr=0.001
    ... )
    >>> model.fit(dataset)
    >>>
    >>> # With lazy initialization
    >>> model = DeepHaloFeature(H=4, L=3, embed_dim=128)
    >>> model.fit(dataset)  # n_items and input_dim inferred
    """
    
    def __init__(
        self,
        n_items=None,
        input_dim=None,
        H=4,
        L=3,
        embed_dim=128,
        dropout=0.0,
        optimizer="Adam",
        lr=0.001,
        epochs=500,
        batch_size=1024,
        **kwargs
    ):
        """Initialize DeepHalo Feature-Based model."""
        # Initialize parent class
        super().__init__(
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        
        self.n_items = n_items
        self.input_dim = input_dim
        self.H = H
        self.L = L
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        self.instantiated = False
        
        # Build the model if dimensions are provided
        if n_items is not None and input_dim is not None:
            self._initialize_layers()
            self._build_model()
            self.instantiated = True
    
    def instantiate(self, n_items, input_dim):
        """Instantiate the model from data dimensions.
        
        Parameters
        ----------
        n_items : int
            Number of items in the choice set
        input_dim : int
            Dimension of input features per item
            
        Returns
        -------
        tuple of (dict, list)
            (indexes, weights) where indexes is empty dict and weights is list of trainable weights
        """
        if not self.instantiated:
            self.n_items = n_items
            self.input_dim = input_dim
            print(f"DeepHaloFeature instantiated with n_items={self.n_items}, "
                  f"input_dim={self.input_dim} inferred from data.")
            
            # Initialize layers and build model
            self._initialize_layers()
            self._build_model()
            self.instantiated = True
        
        return {}, self.trainable_weights
    
    def fit(self, choice_dataset, **kwargs):
        """Fit the model to estimate parameters.
        
        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Choice dataset to use for training
        **kwargs
            Additional arguments passed to parent fit method
            
        Returns
        -------
        dict
            Dictionary with fit history
        """
        if not self.instantiated:
            # Infer dimensions from dataset
            n_items = choice_dataset.get_n_items()
            input_dim = choice_dataset.get_n_items_features()
            
            if input_dim == 0:
                raise ValueError("DeepHaloFeature requires items_features_by_choice in the dataset")
            
            self.instantiate(n_items, input_dim)
        
        return super().fit(choice_dataset=choice_dataset, **kwargs)
    
    def _initialize_layers(self):
        """Initialize all layers of the model."""
        # Basic encoder: 3-layer MLP
        self.basic_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed_dim, activation='relu', use_bias=True),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.embed_dim, activation='relu', use_bias=True),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.embed_dim, use_bias=True)
        ])
        self.enc_norm = tf.keras.layers.LayerNormalization()
        
        # Aggregate linear layers (L layers)
        self.aggregate_linears = [
            tf.keras.layers.Dense(self.H, use_bias=True) for _ in range(self.L)
        ]
        
        # Nonlinear transformation layers (L layers)
        self.nonlinear_transforms = [
            NonlinearTransformation(self.H, self.embed_dim, self.dropout_rate) 
            for _ in range(self.L)
        ]
        
        # Final linear layer
        self.final_linear = tf.keras.layers.Dense(1, use_bias=True)
        
        # Optional quality linear layers (not used in forward but in reference)
        # Keeping for compatibility
        self.qualinear1 = tf.keras.layers.Dense(self.embed_dim, use_bias=True)
        self.qualinear2 = tf.keras.layers.Dense(self.embed_dim, use_bias=True)
    
    def _build_model(self):
        """Build the model by doing a forward pass to initialize all weights."""
        # Create dummy input to initialize layers
        dummy_features = tf.zeros((1, self.n_items, self.input_dim))
        dummy_availability = tf.ones((1, self.n_items))
        _ = self._forward(dummy_features, dummy_availability)
    
    def _forward(self, X, availability, training=False):
        """Forward pass through the network.
        
        Parameters
        ----------
        X : tf.Tensor
            Item features of shape (batch_size, n_items, input_dim)
        availability : tf.Tensor
            Availability mask of shape (batch_size, n_items)
        training : bool, optional
            Whether in training mode, by default False
            
        Returns
        -------
        tuple of (tf.Tensor, tf.Tensor)
            (probabilities, logits) both of shape (batch_size, n_items)
        """
        B = tf.shape(X)[0]
        n = tf.shape(X)[1]
        
        # Compute lengths (number of available items per choice)
        lengths = tf.reduce_sum(availability, axis=1)  # (B,)
        
        # Basic encoding
        Z = self.basic_encoder(X, training=training)  # (B, n, embed_dim)
        Z = self.enc_norm(Z)
        X_enc = Z  # Keep encoded features for nonlinear transformation
        
        # Residual layers
        for agg_linear, nonlinear_tf in zip(self.aggregate_linears, self.nonlinear_transforms):
            # Global aggregation
            # Apply linear to Z: (B, n, embed_dim) -> (B, n, H)
            Z_agg = agg_linear(Z)
            
            # Sum over items and divide by number of available items
            # Z_bar: (B, H)
            Z_bar = tf.reduce_sum(Z_agg, axis=1) / tf.expand_dims(lengths, 1)
            # Reshape to (B, 1, H, 1) for broadcasting
            Z_bar = tf.reshape(Z_bar, [B, 1, self.H, 1])
            
            # Nonlinear transformation on X_enc (not Z)
            # phi: (B, n, H, embed_dim)
            phi = nonlinear_tf(X_enc, training=training)
            
            # Apply availability mask
            # availability: (B, n) -> (B, n, 1, 1)
            availability_mask = tf.reshape(availability, [B, n, 1, 1])
            phi = phi * availability_mask
            
            # Element-wise multiply with Z_bar and sum over H
            # (B, n, H, embed_dim) * (B, 1, H, 1) -> (B, n, H, embed_dim)
            phi_aggregated = phi * Z_bar
            # Sum over H dimension and normalize
            phi_aggregated = tf.reduce_sum(phi_aggregated, axis=2) / tf.cast(self.H, tf.float32)  # (B, n, embed_dim)
            
            # Residual connection
            Z = phi_aggregated + Z
        
        # Final linear projection to logits
        logits = self.final_linear(Z)  # (B, n, 1)
        logits = tf.squeeze(logits, axis=-1)  # (B, n)
        
        # Apply availability mask with -inf for unavailable items
        mask = tf.equal(availability, 1.0)
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), float('-inf')))
        
        # Compute probabilities
        probas = tf.nn.softmax(logits, axis=-1)
        
        return probas, logits
    
    @property
    def trainable_weights(self):
        """Return list of all trainable weights.
        
        Returns
        -------
        list of tf.Variable
            All trainable parameters in the model
        """
        weights = []
        
        # Basic encoder weights
        weights.extend(self.basic_encoder.trainable_variables)
        weights.extend(self.enc_norm.trainable_variables)
        
        # Aggregate linear layers
        for layer in self.aggregate_linears:
            weights.extend(layer.trainable_variables)
        
        # Nonlinear transformation layers
        for layer in self.nonlinear_transforms:
            weights.extend(layer.trainable_variables)
        
        # Final linear layer
        weights.extend(self.final_linear.trainable_variables)
        
        # Quality linear layers (if used)
        weights.extend(self.qualinear1.trainable_variables)
        weights.extend(self.qualinear2.trainable_variables)
        
        return weights
    
    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utilities from item features.
        
        Parameters
        ----------
        shared_features_by_choice : tf.Tensor or np.ndarray
            Shared features (not used by this model)
        items_features_by_choice : tuple of tf.Tensor or np.ndarray
            Items features - tuple of (items_features, ) where items_features has shape
            (batch_size, n_items, n_features)
        available_items_by_choice : tf.Tensor or np.ndarray
            Binary availability mask of shape (batch_size, n_items)
        choices : tf.Tensor or np.ndarray
            Actual choices (not used in utility computation)
            
        Returns
        -------
        tf.Tensor
            Utilities of shape (batch_size, n_items)
            
        Notes
        -----
        Returns logits as utilities because the ChoiceModel base class will apply
        softmax to convert utilities to probabilities.
        """
        # Unused parameters
        _ = shared_features_by_choice
        _ = choices
        
        # Extract item features
        # items_features_by_choice is a tuple: (items_features, contexts_items_features)
        # We want items_features
        if isinstance(items_features_by_choice, tuple):
            item_features = items_features_by_choice[0]
        else:
            item_features = items_features_by_choice
        
        # Convert to float32
        item_features = tf.cast(item_features, tf.float32)
        availability = tf.cast(available_items_by_choice, tf.float32)
        
        # Get logits from forward pass
        _, logits = self._forward(item_features, availability, training=True)
        
        return logits
    
    def save_model(self, path, save_opt=True):
        """Save the model to disk.
        
        Parameters
        ----------
        path : str
            Path to the folder where to save the model
        save_opt : bool, optional
            Whether to save optimizer state, by default True
        """
        # Save using parent method which handles weights and basic params
        super().save_model(path, save_opt=save_opt)
        
        # The aggregate_linears and nonlinear_transforms lists are not saved by parent
        # because they contain non-serializable layer objects
        # The weights are already saved by parent's save_model
        # When loading, we reconstruct these lists in _initialize_layers
    
    @classmethod
    def load_model(cls, path):
        """Load a DeepHaloFeature model previously saved with save_model().
        
        Parameters
        ----------
        path : str
            Path to the folder where the saved model files are
            
        Returns
        -------
        DeepHaloFeature
            Loaded DeepHaloFeature model
        """
        # Load parameters from params.json
        with open(Path(path) / "params.json") as f:
            params = json.load(f)
        
        # Create model instance
        obj = cls(
            n_items=params.get('n_items'),
            input_dim=params.get('input_dim'),
            H=params.get('H', 4),
            L=params.get('L', 3),
            embed_dim=params.get('embed_dim', 128),
            dropout=params.get('dropout_rate', 0.0),
            optimizer=params['optimizer_name'],
            lr=params.get('lr', 0.001),
            epochs=params.get('epochs', 500),
            batch_size=params.get('batch_size', 1024),
        )
        
        # Load weights
        loaded_weights = []
        i = 0
        weight_path = f"weight_{i}.npy"
        files_list = [str(file.name) for file in Path(path).iterdir()]
        
        while weight_path in files_list:
            loaded_weights.append(np.load(Path(path) / weight_path))
            i += 1
            weight_path = f"weight_{i}.npy"
        
        # Assign loaded weights
        # Order must match trainable_weights property
        weight_idx = 0
        
        # Basic encoder
        for var in obj.basic_encoder.trainable_variables:
            var.assign(loaded_weights[weight_idx])
            weight_idx += 1
        
        # Enc norm
        for var in obj.enc_norm.trainable_variables:
            var.assign(loaded_weights[weight_idx])
            weight_idx += 1
        
        # Aggregate linears
        for layer in obj.aggregate_linears:
            for var in layer.trainable_variables:
                var.assign(loaded_weights[weight_idx])
                weight_idx += 1
        
        # Nonlinear transforms
        for layer in obj.nonlinear_transforms:
            for var in layer.trainable_variables:
                var.assign(loaded_weights[weight_idx])
                weight_idx += 1
        
        # Final linear
        for var in obj.final_linear.trainable_variables:
            var.assign(loaded_weights[weight_idx])
            weight_idx += 1
        
        # Quality linears
        for var in obj.qualinear1.trainable_variables:
            var.assign(loaded_weights[weight_idx])
            weight_idx += 1
        for var in obj.qualinear2.trainable_variables:
            var.assign(loaded_weights[weight_idx])
            weight_idx += 1
        
        # Set other attributes
        for k, v in params.items():
            if k not in ['n_items', 'input_dim', 'H', 'L', 'embed_dim', 
                         'dropout_rate', 'optimizer_name', 'lr', 'epochs', 'batch_size']:
                setattr(obj, k, v)
        
        # Load optimizer state if available
        if Path.is_dir(Path(path) / "optimizer"):
            with open(Path(path) / "optimizer" / "config.json") as f:
                config = json.load(f)
            obj.optimizer = obj.optimizer.from_config(config)
            obj.optimizer.build(var_list=obj.trainable_weights)
            
            with open(Path(path) / "optimizer" / "weights_store.json") as f:
                store = json.load(f)
            for key, value in store.items():
                store[key] = np.array(value, dtype=np.float32)
            obj.optimizer.load_own_variables(store)
        
        return obj
