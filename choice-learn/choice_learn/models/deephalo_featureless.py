"""DeepHalo Featureless Model

This module implements a featureless deep neural network model for discrete choice prediction.
The model uses residual blocks (Quadratic or Exact) to learn complex choice patterns (including halo effects)
from availability masks only, without requiring item features.

Note: This module uses 'batch_size' terminology for consistency with variable naming conventions,
though it is equivalent to 'n_choices' used in ChoiceModel base class comments. Both refer to
the number of choice samples (decisions) in a batch.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .base_model import ChoiceModel


class QuaResBlock(tf.keras.layers.Layer):
    """Quadratic Residual Block.
    
    Applies a quadratic transformation: linear(x^2) + x
    """
    
    def __init__(self, d, **kwargs):
        """Initialize QuaResBlock.
        
        Parameters
        ----------
        d : int
            Dimension of the block
        """
        super().__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(d, use_bias=False)
    
    def call(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor
            
        Returns
        -------
        tf.Tensor
            Output after quadratic residual transformation
        """
        return self.linear(tf.math.pow(x, 2)) + x


class ExaResBlock(tf.keras.layers.Layer):
    """Exact Residual Block.
    
    Applies element-wise multiplication with learned activations.
    """
    
    def __init__(self, input_dim, hidden_dim, **kwargs):
        """Initialize ExaResBlock.
        
        Parameters
        ----------
        input_dim : int
            Input dimension (original availability vector size)
        hidden_dim : int
            Hidden dimension for transformations
        """
        super().__init__(**kwargs)
        self.linear_main = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.linear_act = tf.keras.layers.Dense(hidden_dim, use_bias=False)
    
    def call(self, inputs):
        """Forward pass.
        
        Parameters
        ----------
        inputs : tuple of (tf.Tensor, tf.Tensor)
            (z_prev, z0) where z_prev is the current state and z0 is the original input
            
        Returns
        -------
        tf.Tensor
            Output after exact residual transformation
        """
        z_prev, z0 = inputs
        return self.linear_main(z_prev * self.linear_act(z0)) + z_prev


class DeepHaloFeatureless(ChoiceModel):
    """DeepHalo Featureless Model.
    
    A deep neural network for discrete choice modeling that operates purely on item
    availability masks without requiring item or customer features. Uses residual blocks
    to learn complex choice patterns.
    
    Parameters
    ----------
    opt_size : int
        Size of the choice set (number of items)
    depth : int
        Number of residual blocks + 1
    resnet_width : int
        Hidden dimension of the residual network
    block_types : list of str
        List of block types, each must be 'qua' (quadratic) or 'exa' (exact).
        Length must be depth - 1.
    optimizer : str, optional
        Optimizer to use ('lbfgs', 'Adam', 'SGD', etc.), by default 'Adam'
    lr : float, optional
        Learning rate for optimizer, by default 0.001
    epochs : int, optional
        Maximum number of training epochs, by default 1000
    batch_size : int, optional
        Batch size for stochastic optimizers, by default 32
    loss_type : str, optional
        Loss function type. Options:
        - 'nll' or 'cross_entropy': Negative Log-Likelihood (default, recommended)
        - 'mse': Mean Squared Error (experimental, for comparison)
        By default 'mse'
    **kwargs
        Additional arguments passed to ChoiceModel base class
        
    Raises
    ------
    ValueError
        If length of block_types doesn't match depth - 1
    ValueError
        If unknown block type is specified
    ValueError
        If unknown loss_type is specified
        
    Example
    -------
    >>> # Using MSE (experimental)
    >>> model_mse = DeepHaloFeatureless(
    ...     opt_size=20,
    ...     depth=5,
    ...     resnet_width=64,
    ...     block_types=['qua'] * 4,
    ...     optimizer='Adam',
    ...     lr=0.001,
    ...     loss_type='mse'
    ... )
    >>> model.fit(dataset)
    """
    
    def __init__(
        self,
        opt_size,
        depth,
        resnet_width,
        block_types,
        optimizer="Adam",
        lr=0.001,
        epochs=500,
        batch_size=1024,
        loss_type="mse",
        **kwargs
    ):
        """Initialize DeepHalo Featureless model.
        
        Parameters
        ----------
        loss_type : str, optional
            Type of loss function to use. Options:
            - 'mse': Mean Squared Error (default)
            - 'nll' or 'cross_entropy': Negative Log-Likelihood (alternative)
        """
        # Store loss_type before calling parent init
        self.loss_type = loss_type.lower()
        
        # Initialize parent class (don't pass loss_type to it)
        super().__init__(
            optimizer=optimizer,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        
        if len(block_types) != depth - 1:
            raise ValueError(
                f"Length of block_types ({len(block_types)}) must equal depth - 1 ({depth - 1})"
            )
        
        self.opt_size = opt_size
        self.depth = depth
        self.resnet_width = resnet_width
        self.block_types = block_types
        
        # Set custom loss function if MSE is selected (after parent init)
        if self.loss_type in ['mse', 'mean_squared_error']:
            self.loss = tf.keras.losses.MeanSquaredError()
            print("Using MSE loss for DeepHalo featureless discrete choice model. ")
        elif self.loss_type in ['nll', 'cross_entropy', 'categorical_crossentropy']:
            # Use default NLL from base class (already set by parent __init__)
            pass
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. "
                           f"Choose from 'nll', 'cross_entropy', or 'mse'.")
        
        # Initialize layers
        self.in_lin = tf.keras.layers.Dense(resnet_width, use_bias=False)
        self.out_lin = tf.keras.layers.Dense(opt_size, use_bias=False)
        self.blocks = []
        
        for t in block_types:
            if t == "exa":
                self.blocks.append(ExaResBlock(opt_size, resnet_width))
            elif t == "qua":
                self.blocks.append(QuaResBlock(resnet_width))
            else:
                raise ValueError(f"Unknown block type: {t}. Must be 'exa' or 'qua'")
        
        # Build the model to initialize weights
        self._build_model()
    
    def _build_model(self):
        """Build the model by doing a forward pass to initialize all weights."""
        # Create dummy input to initialize layers
        dummy_input = tf.zeros((1, self.opt_size))
        _ = self._forward(dummy_input)
    
    def _forward(self, e):
        """Forward pass through the network.
        
        Parameters
        ----------
        e : tf.Tensor
            Availability mask of shape (batch_size, opt_size)
            
        Returns
        -------
        tuple of (tf.Tensor, tf.Tensor)
            (probabilities, logits) both of shape (batch_size, opt_size)
        """
        mask = tf.equal(e, 1.0)
        e0 = tf.identity(e)
        e = self.in_lin(e)
        
        for b in self.blocks:
            if isinstance(b, ExaResBlock):
                e = b([e, e0])
            else:
                e = b(e)
        
        logits = self.out_lin(e)
        # Mask out unavailable items with -inf
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), float("-inf")))
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
        weights.extend(self.in_lin.trainable_variables)
        weights.extend(self.out_lin.trainable_variables)
        for block in self.blocks:
            weights.extend(block.trainable_variables)
        return weights
    
    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ):
        """Compute utilities from availability masks.
        
        This model uses the available_items_by_choice mask as input.
        If items_features_by_choice is provided and is the availability mask,
        it will be used instead.
        
        Parameters
        ----------
        shared_features_by_choice : tf.Tensor or np.ndarray
            Shared features (not used by this model)
        items_features_by_choice : tf.Tensor or np.ndarray or tuple
            Items features - expected to be availability mask
        available_items_by_choice : tf.Tensor or np.ndarray
            Binary availability mask of shape (batch_size, n_items)
            where 1 indicates available and 0 indicates unavailable
        choices : tf.Tensor or np.ndarray
            Actual choices (not used in utility computation)
            
        Returns
        -------
        tf.Tensor
            Utilities of shape (batch_size, n_items)
            
        Notes
        -----
        This method returns logits (not probabilities) as utilities because the
        ChoiceModel base class will apply softmax to convert utilities to probabilities.
        """
        # Unused parameters
        _ = shared_features_by_choice
        _ = choices
        
        # Get availability mask from items_features_by_choice
        # items_features_by_choice should be (batch_size, n_items, 1) with availability in first feature
        if items_features_by_choice is not None and not isinstance(items_features_by_choice, tuple):
            # items_features_by_choice is (batch_size, n_items, n_features)
            # We need (batch_size, n_items)
            items_features_by_choice = tf.cast(items_features_by_choice, tf.float32)
            if len(items_features_by_choice.shape) == 3:
                # Take first feature dimension (availability)
                availability = items_features_by_choice[:, :, 0]
            elif len(items_features_by_choice.shape) == 2:
                availability = items_features_by_choice
            else:
                raise ValueError(f"Unexpected items_features_by_choice shape: {items_features_by_choice.shape}")
        else:
            # Fallback to available_items_by_choice
            availability = tf.cast(available_items_by_choice, tf.float32)
        
        # Get logits from forward pass
        _, logits = self._forward(availability)
        
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
        super().save_model(path, save_opt=save_opt)
        
        # The base class save_model already saves most attributes through params.json
        # No additional saving needed as opt_size, depth, resnet_width, block_types, loss_type
        # are already captured in the params.json by the base class
    
    @classmethod
    def load_model(cls, path):
        """Load a DeepHaloFeatureless model previously saved with save_model().
        
        Parameters
        ----------
        path : str
            Path to the folder where the saved model files are
            
        Returns
        -------
        DeepHaloFeatureless
            Loaded DeepHaloFeatureless model
        """
        # Load parameters from params.json
        with open(Path(path) / "params.json") as f:
            params = json.load(f)
        
        # Extract required initialization parameters
        obj = cls(
            opt_size=params['opt_size'],
            depth=params['depth'],
            resnet_width=params['resnet_width'],
            block_types=params['block_types'],
            optimizer=params['optimizer_name'],
            lr=params.get('lr', 0.001),
            epochs=params.get('epochs', 1000),
            batch_size=params.get('batch_size', 32),
            loss_type=params.get('loss_type', 'mse'),
        )
        
        # Load weights from files
        loaded_weights = []
        i = 0
        weight_path = f"weight_{i}.npy"
        files_list = [str(file.name) for file in Path(path).iterdir()]
        
        while weight_path in files_list:
            loaded_weights.append(np.load(Path(path) / weight_path))
            i += 1
            weight_path = f"weight_{i}.npy"
        
        # Assign loaded weights to the model's layers
        # IMPORTANT: Order must match trainable_weights property:
        # in_lin weights, then out_lin weights, then blocks weights
        weight_idx = 0
        
        # Assign to in_lin
        for var in obj.in_lin.trainable_variables:
            var.assign(loaded_weights[weight_idx])
            weight_idx += 1
        
        # Assign to out_lin
        for var in obj.out_lin.trainable_variables:
            var.assign(loaded_weights[weight_idx])
            weight_idx += 1
        
        # Assign to blocks
        for block in obj.blocks:
            for var in block.trainable_variables:
                var.assign(loaded_weights[weight_idx])
                weight_idx += 1
        
        # Set other attributes from params
        for k, v in params.items():
            if k not in ['opt_size', 'depth', 'resnet_width', 'block_types', 
                         'optimizer_name', 'lr', 'epochs', 'batch_size', 'loss_type']:
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
