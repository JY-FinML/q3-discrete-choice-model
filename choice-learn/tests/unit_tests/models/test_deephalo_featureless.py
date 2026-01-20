"""Unit tests for the DeepHaloFeatureless model."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models import DeepHaloFeatureless
import choice_learn.tf_ops as tf_ops


# Create small test dataset
test_dataset = ChoiceDataset(
    items_features_by_choice=np.array([
        [[1.0], [1.0], [1.0]],
        [[1.0], [1.0], [1.0]],
        [[1.0], [1.0], [1.0]],
        [[1.0], [1.0], [0.0]],
    ]).astype('float32'),
    available_items_by_choice=np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ]).astype('float32'),
    choices=np.array([0, 1, 2, 0]),
)


def test_deephalo_instantiation_qua():
    """Tests DeepHaloFeatureless instantiation with quadratic blocks."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        loss_type='nll'
    )
    
    assert model.n_items == 3
    assert model.depth == 3
    assert model.hidden_dim == 16
    assert len(model.blocks) == 2
    assert len(model.trainable_weights) > 0
    assert model.instantiated == True


def test_deephalo_instantiation_exa():
    """Tests DeepHaloFeatureless instantiation with exact blocks."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['exa', 'exa'],
        optimizer='Adam',
        lr=0.0001,
        loss_type='nll'
    )
    
    assert model.n_items == 3
    assert model.depth == 3
    assert len(model.blocks) == 2
    assert len(model.trainable_weights) > 0


def test_deephalo_instantiation_mixed():
    """Tests DeepHaloFeatureless instantiation with mixed block types."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=4,
        hidden_dim=16,
        block_types=['qua', 'exa', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        loss_type='nll'
    )
    
    assert len(model.blocks) == 3
    assert model.block_types == ['qua', 'exa', 'qua']


def test_deephalo_invalid_block_type():
    """Tests that invalid block type raises error."""
    with pytest.raises(ValueError, match="Unknown block type"):
        DeepHaloFeatureless(
            n_items=3,
            depth=3,
            hidden_dim=16,
            block_types=['invalid', 'qua'],
            optimizer='Adam',
            lr=0.0001
        )


def test_deephalo_invalid_depth():
    """Tests that mismatched depth and block_types raises error."""
    with pytest.raises(ValueError, match="Length of block_types"):
        DeepHaloFeatureless(
            n_items=3,
            depth=5,
            hidden_dim=16,
            block_types=['qua', 'qua'],  # Should have 4 blocks for depth=5
            optimizer='Adam',
            lr=0.0001
        )


def test_deephalo_invalid_loss_type():
    """Tests that invalid loss type raises error."""
    with pytest.raises(ValueError, match="Unknown loss_type"):
        DeepHaloFeatureless(
            n_items=3,
            depth=3,
            hidden_dim=16,
            block_types=['qua', 'qua'],
            optimizer='Adam',
            lr=0.0001,
            loss_type='invalid_loss'
        )


def test_deephalo_loss_type_nll():
    """Tests that NLL loss type is correctly set."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        loss_type='nll'
    )
    
    assert model.loss_type == 'nll'
    assert isinstance(model.loss, tf_ops.CustomCategoricalCrossEntropy)


def test_deephalo_loss_type_mse():
    """Tests that MSE loss type is correctly set."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        loss_type='mse'
    )
    
    assert model.loss_type == 'mse'
    assert isinstance(model.loss, tf.keras.losses.MeanSquaredError)


def test_forward_pass():
    """Tests forward pass through the network."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001
    )
    
    # Test with simple availability mask
    availability = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=tf.float32)
    probas, logits = model._forward(availability)
    
    assert probas.shape == (2, 3)
    assert logits.shape == (2, 3)
    
    # Check that probabilities sum to 1
    assert np.allclose(tf.reduce_sum(probas, axis=-1).numpy(), 1.0, atol=1e-5)
    
    # Check that unavailable items have 0 probability
    assert probas[1, 2].numpy() == 0.0


def test_compute_batch_utility():
    """Tests compute_batch_utility method."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001
    )
    
    utilities = model.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=np.array([
            [[1.0], [1.0], [1.0]],
            [[1.0], [1.0], [0.0]]
        ]).astype('float32'),
        available_items_by_choice=np.array([[1, 1, 1], [1, 1, 0]]).astype('float32'),
        choices=np.array([0, 1])
    )
    
    assert utilities.shape == (2, 3)
    # Logits for unavailable items should be -inf
    assert tf.math.is_inf(utilities[1, 2])


def test_trainable_weights():
    """Tests that trainable_weights returns all model parameters."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=5,
        depth=4,
        hidden_dim=32,
        block_types=['qua', 'exa', 'qua'],
        optimizer='Adam',
        lr=0.0001
    )
    
    weights = model.trainable_weights
    
    # Should have weights from: in_lin, out_lin, and 3 blocks
    assert len(weights) > 0
    
    # Check that all are TensorFlow variables (use hasattr to check for Variable attributes)
    for w in weights:
        assert hasattr(w, 'numpy') and hasattr(w, 'assign')


def test_fit_with_nll():
    """Tests that model can fit with NLL loss."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=5,
        batch_size=2,
        loss_type='nll'
    )
    
    nll_before = model.evaluate(test_dataset)
    history = model.fit(test_dataset)
    nll_after = model.evaluate(test_dataset)
    
    # Loss should decrease (or at least not increase significantly)
    assert nll_after <= nll_before + 0.1  # Allow small tolerance
    assert 'train_loss' in history


def test_fit_with_mse():
    """Tests that model can fit with MSE loss."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=5,
        batch_size=2,
        loss_type='mse'
    )
    
    loss_before = model.evaluate(test_dataset)
    history = model.fit(test_dataset)
    loss_after = model.evaluate(test_dataset)
    
    # Loss should decrease
    assert loss_after <= loss_before + 0.1
    assert 'train_loss' in history


def test_predict_probas():
    """Tests that predict_probas returns valid probabilities."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001
    )
    
    probas = model.predict_probas(test_dataset)
    
    assert probas.shape == (4, 3)
    
    # Check that probabilities sum to 1
    assert np.allclose(np.sum(probas, axis=-1), 1.0, atol=1e-5)
    
    # Check that all probabilities are in [0, 1]
    assert np.all(probas >= 0.0)
    assert np.all(probas <= 1.0)
    
    # Check that unavailable items have 0 probability
    assert probas[3, 2] == 0.0


def test_evaluate():
    """Tests that evaluate returns a valid loss value."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        loss_type='nll'
    )
    
    loss = model.evaluate(test_dataset)
    
    # Loss should be a positive number
    # Convert to Python float if it's a tensor
    if hasattr(loss, 'numpy'):
        loss = float(loss.numpy())
    assert isinstance(loss, (float, np.floating))
    assert loss > 0.0


def test_different_optimizers():
    """Tests that model works with different optimizers."""
    tf.config.run_functions_eagerly(True)
    
    for optimizer in ['Adam', 'SGD', 'RMSprop']:
        model = DeepHaloFeatureless(
            n_items=3,
            depth=3,
            hidden_dim=16,
            block_types=['qua', 'qua'],
            optimizer=optimizer,
            lr=0.0001,
            epochs=2,
            batch_size=2
        )
        
        history = model.fit(test_dataset)
        assert 'train_loss' in history


def test_batch_size_handling():
    """Tests that model handles different batch sizes correctly."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001
    )
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4, -1]:
        loss = model.evaluate(test_dataset, batch_size=batch_size)
        assert loss > 0.0


def test_items_features_shape_2d():
    """Tests that model handles 2D items_features correctly."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        n_items=3,
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001
    )
    
    # Test with 2D availability (batch_size, n_items)
    utilities = model.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0]
        ]).astype('float32'),
        available_items_by_choice=np.array([[1, 1, 1], [1, 1, 0]]).astype('float32'),
        choices=np.array([0, 1])
    )
    
    assert utilities.shape == (2, 3)


def test_lazy_instantiation():
    """Tests that model can be instantiated without n_items."""
    tf.config.run_functions_eagerly(True)
    
    # Create model without n_items
    model = DeepHaloFeatureless(
        depth=3,
        hidden_dim=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=2,
        loss_type='nll'
    )
    
    assert model.n_items is None
    assert model.instantiated == False
    assert not hasattr(model, 'in_lin')  # Layers not created yet
    
    # Fit should trigger instantiation
    model.fit(test_dataset)
    
    assert model.n_items == 3  # Inferred from dataset
    assert model.instantiated == True
    assert hasattr(model, 'in_lin')
    assert len(model.trainable_weights) > 0


def test_instantiate_method():
    """Tests the instantiate() method directly."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        depth=4,
        hidden_dim=32,
        block_types=['qua', 'exa', 'qua'],
        optimizer='Adam',
        lr=0.0001
    )
    
    assert model.instantiated == False
    
    # Call instantiate manually
    indexes, weights = model.instantiate(n_items=5)
    
    assert model.instantiated == True
    assert model.n_items == 5
    assert indexes == {}  # Featureless model returns empty dict
    assert len(weights) > 0
    assert len(weights) == len(model.trainable_weights)


def test_default_block_types():
    """Tests that default block_types are created when not specified."""
    tf.config.run_functions_eagerly(True)
    
    # Create model with only depth, block_types should default to all 'qua'
    model = DeepHaloFeatureless(
        n_items=4,
        depth=5,
        hidden_dim=16,
        optimizer='Adam',
        lr=0.0001
    )
    
    assert model.block_types == ['qua', 'qua', 'qua', 'qua']
    assert len(model.blocks) == 4
    assert model.instantiated == True


def test_instantiate_repeatable():
    """Tests that calling instantiate() multiple times doesn't re-initialize."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        depth=3,
        hidden_dim=16,
        optimizer='Adam',
        lr=0.0001
    )
    
    # First instantiation
    model.instantiate(n_items=3)
    weights_before = [w.numpy().copy() for w in model.trainable_weights]
    
    # Second instantiation should not re-initialize
    model.instantiate(n_items=3)
    weights_after = [w.numpy() for w in model.trainable_weights]
    
    # Weights should be the same
    for w_before, w_after in zip(weights_before, weights_after):
        assert np.allclose(w_before, w_after)
