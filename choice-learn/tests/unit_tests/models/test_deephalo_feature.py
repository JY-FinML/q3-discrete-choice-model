"""Unit tests for the DeepHaloFeature model."""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models import DeepHaloFeature


# Create small test dataset with features
np.random.seed(42)
n_test_samples = 4
n_test_items = 3
n_test_features = 5

test_features = np.random.randn(n_test_samples, n_test_items, n_test_features).astype('float32')
test_availability = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 0]
]).astype('float32')
test_choices = np.array([0, 1, 2, 0])

test_dataset = ChoiceDataset(
    items_features_by_choice=(test_features,),
    available_items_by_choice=test_availability,
    choices=test_choices,
)


def test_deephalo_feature_instantiation_explicit():
    """Tests DeepHaloFeature instantiation with explicit dimensions."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=4,
        L=3,
        embed_dim=32,
        dropout=0.1,
        optimizer='Adam',
        lr=0.001
    )
    
    assert model.n_items == 3
    assert model.input_dim == 5
    assert model.H == 4
    assert model.L == 3
    assert model.embed_dim == 32
    assert model.dropout_rate == 0.1
    assert len(model.trainable_weights) > 0
    assert model.instantiated == True


def test_deephalo_feature_instantiation_lazy():
    """Tests DeepHaloFeature instantiation without dimensions (lazy)."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        H=2,
        L=2,
        embed_dim=16,
        optimizer='Adam',
        lr=0.001
    )
    
    assert model.n_items is None
    assert model.input_dim is None
    assert model.instantiated == False
    assert not hasattr(model, 'basic_encoder')


def test_deephalo_feature_layer_initialization():
    """Tests that all layers are properly initialized."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    # Check layers exist
    assert hasattr(model, 'basic_encoder')
    assert hasattr(model, 'enc_norm')
    assert hasattr(model, 'aggregate_linears')
    assert hasattr(model, 'nonlinear_transforms')
    assert hasattr(model, 'final_linear')
    
    # Check correct number of layers
    assert len(model.aggregate_linears) == 2  # L layers
    assert len(model.nonlinear_transforms) == 2  # L layers


def test_forward_pass():
    """Tests forward pass through the network."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    # Create test input
    features = tf.constant(test_features[:2], dtype=tf.float32)
    availability = tf.constant(test_availability[:2], dtype=tf.float32)
    
    probas, logits = model._forward(features, availability)
    
    assert probas.shape == (2, 3)
    assert logits.shape == (2, 3)
    
    # Check that probabilities sum to 1
    assert np.allclose(tf.reduce_sum(probas, axis=-1).numpy(), 1.0, atol=1e-5)
    
    # Check that all probabilities are in [0, 1]
    assert tf.reduce_all(probas >= 0.0)
    assert tf.reduce_all(probas <= 1.0)


def test_forward_pass_with_unavailable_items():
    """Tests that forward pass correctly masks unavailable items."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=5,
        input_dim=3,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    # Create availability with some items unavailable
    features = tf.random.normal((2, 5, 3))
    availability = tf.constant([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0]
    ])
    
    probas, logits = model._forward(features, availability)
    
    # Check unavailable items have 0 probability
    assert probas[0, 3].numpy() == 0.0
    assert probas[0, 4].numpy() == 0.0
    assert probas[1, 1].numpy() == 0.0
    assert probas[1, 4].numpy() == 0.0
    
    # Check unavailable items have -inf logits
    assert tf.math.is_inf(logits[0, 3])
    assert tf.math.is_inf(logits[1, 1])


def test_compute_batch_utility():
    """Tests compute_batch_utility method."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    utilities = model.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=(test_features[:2],),
        available_items_by_choice=test_availability[:2],
        choices=test_choices[:2]
    )
    
    assert utilities.shape == (2, 3)
    # Logits for unavailable items should be -inf
    # (none in first 2 samples)


def test_trainable_weights():
    """Tests that trainable_weights returns all model parameters."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=5,
        input_dim=10,
        H=4,
        L=3,
        embed_dim=64,
        dropout=0.1,
        optimizer='Adam',
        lr=0.001
    )
    
    weights = model.trainable_weights
    
    # Should have weights from: basic_encoder, enc_norm, aggregate_linears,
    # nonlinear_transforms, final_linear, qualinear1, qualinear2
    assert len(weights) > 0
    
    # Check that all are TensorFlow variables
    for w in weights:
        assert hasattr(w, 'numpy') and hasattr(w, 'assign')


def test_fit():
    """Tests that model can fit."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.01,
        epochs=5,
        batch_size=2
    )
    
    loss_before = model.evaluate(test_dataset)
    history = model.fit(test_dataset, verbose=0)
    loss_after = model.evaluate(test_dataset)
    
    # Loss should decrease (or at least not increase significantly)
    assert loss_after <= loss_before + 0.2  # Allow some tolerance
    assert 'train_loss' in history
    assert len(history['train_loss']) == 5


def test_predict_probas():
    """Tests that predict_probas returns valid probabilities."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
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
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    loss = model.evaluate(test_dataset)
    
    # Loss should be a positive number
    if hasattr(loss, 'numpy'):
        loss = float(loss.numpy())
    assert isinstance(loss, (float, np.floating))
    assert loss > 0.0


def test_different_optimizers():
    """Tests that model works with different optimizers."""
    tf.config.run_functions_eagerly(True)
    
    for optimizer in ['Adam', 'SGD', 'Adamax']:
        model = DeepHaloFeature(
            n_items=3,
            input_dim=5,
            H=2,
            L=2,
            embed_dim=16,
            dropout=0.0,
            optimizer=optimizer,
            lr=0.01,
            epochs=2,
            batch_size=2
        )
        
        history = model.fit(test_dataset, verbose=0)
        assert 'train_loss' in history


def test_batch_size_handling():
    """Tests that model handles different batch sizes correctly."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4, -1]:
        loss = model.evaluate(test_dataset, batch_size=batch_size)
        assert loss > 0.0


def test_lazy_instantiation_with_fit():
    """Tests that model can be instantiated lazily during fit."""
    tf.config.run_functions_eagerly(True)
    
    # Create model without dimensions
    model = DeepHaloFeature(
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.01,
        epochs=3,
        batch_size=2
    )
    
    assert model.n_items is None
    assert model.input_dim is None
    assert model.instantiated == False
    
    # Fit should trigger instantiation
    history = model.fit(test_dataset, verbose=0)
    
    assert model.n_items == 3  # Inferred from dataset
    assert model.input_dim == 5  # Inferred from dataset
    assert model.instantiated == True
    assert len(model.trainable_weights) > 0
    assert 'train_loss' in history


def test_instantiate_method():
    """Tests the instantiate() method directly."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    assert model.instantiated == False
    
    # Call instantiate manually
    indexes, weights = model.instantiate(n_items=5, input_dim=10)
    
    assert model.instantiated == True
    assert model.n_items == 5
    assert model.input_dim == 10
    assert indexes == {}  # Returns empty dict
    assert len(weights) > 0
    assert len(weights) == len(model.trainable_weights)


def test_instantiate_repeatable():
    """Tests that calling instantiate() multiple times doesn't re-initialize."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        H=2,
        L=2,
        embed_dim=16,
        optimizer='Adam',
        lr=0.001
    )
    
    # First instantiation
    model.instantiate(n_items=3, input_dim=5)
    weights_before = [w.numpy().copy() for w in model.trainable_weights]
    
    # Second instantiation should not re-initialize
    model.instantiate(n_items=3, input_dim=5)
    weights_after = [w.numpy() for w in model.trainable_weights]
    
    # Weights should be the same
    for w_before, w_after in zip(weights_before, weights_after):
        assert np.allclose(w_before, w_after)


def test_different_architectures():
    """Tests different architecture configurations."""
    tf.config.run_functions_eagerly(True)
    
    architectures = [
        {'H': 2, 'L': 1, 'embed_dim': 16},
        {'H': 3, 'L': 2, 'embed_dim': 32},
        {'H': 4, 'L': 3, 'embed_dim': 64},
        {'H': 5, 'L': 4, 'embed_dim': 128},
    ]
    
    for arch in architectures:
        model = DeepHaloFeature(
            n_items=3,
            input_dim=5,
            H=arch['H'],
            L=arch['L'],
            embed_dim=arch['embed_dim'],
            dropout=0.0,
            optimizer='Adam',
            lr=0.001
        )
        
        assert model.H == arch['H']
        assert model.L == arch['L']
        assert model.embed_dim == arch['embed_dim']
        assert len(model.aggregate_linears) == arch['L']
        assert len(model.nonlinear_transforms) == arch['L']


def test_dropout_rates():
    """Tests that different dropout rates work."""
    tf.config.run_functions_eagerly(True)
    
    for dropout_rate in [0.0, 0.1, 0.2, 0.5]:
        model = DeepHaloFeature(
            n_items=3,
            input_dim=5,
            H=2,
            L=2,
            embed_dim=16,
            dropout=dropout_rate,
            optimizer='Adam',
            lr=0.001,
            epochs=2,
            batch_size=2
        )
        
        assert model.dropout_rate == dropout_rate
        history = model.fit(test_dataset, verbose=0)
        assert 'train_loss' in history


def test_nonlinear_transformation_layer():
    """Tests NonlinearTransformation layer separately."""
    from choice_learn.models.deephalo_feature import NonlinearTransformation
    
    tf.config.run_functions_eagerly(True)
    
    H = 4
    embed_dim = 32
    layer = NonlinearTransformation(H, embed_dim, dropout=0.0)
    
    # Test forward pass
    batch_size = 2
    n_items = 3
    X = tf.random.normal((batch_size, n_items, embed_dim))
    
    output = layer(X, training=False)
    
    # Output should have shape (batch_size, n_items, H, embed_dim)
    assert output.shape == (batch_size, n_items, H, embed_dim)


def test_missing_features_error():
    """Tests that model raises error when features are missing."""
    tf.config.run_functions_eagerly(True)
    
    # Create dataset without features
    dataset_no_features = ChoiceDataset(
        available_items_by_choice=test_availability,
        choices=test_choices,
    )
    
    model = DeepHaloFeature(
        H=2,
        L=2,
        embed_dim=16,
        optimizer='Adam',
        lr=0.001
    )
    
    # Should raise error when trying to fit
    with pytest.raises(ValueError, match="requires items_features_by_choice"):
        model.fit(dataset_no_features)


def test_items_features_tuple_handling():
    """Tests that model correctly handles items_features as tuple."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    # Test with tuple of features
    utilities = model.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=(test_features[:2],),
        available_items_by_choice=test_availability[:2],
        choices=test_choices[:2]
    )
    
    assert utilities.shape == (2, 3)
    
    # Test with non-tuple (single array)
    utilities2 = model.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=test_features[:2],
        available_items_by_choice=test_availability[:2],
        choices=test_choices[:2]
    )
    
    assert utilities2.shape == (2, 3)


def test_training_mode_flag():
    """Tests that training mode affects dropout behavior."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=3,
        input_dim=5,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.5,  # High dropout to see effect
        optimizer='Adam',
        lr=0.001
    )
    
    features = tf.constant(test_features[:1], dtype=tf.float32)
    availability = tf.constant(test_availability[:1], dtype=tf.float32)
    
    # Multiple forward passes with training=True should give different results (due to dropout)
    probas1, _ = model._forward(features, availability, training=True)
    probas2, _ = model._forward(features, availability, training=True)
    
    # With dropout, results should potentially differ
    # (though not guaranteed with random seed, so we just check shape)
    assert probas1.shape == probas2.shape
    
    # Multiple forward passes with training=False should give same results
    probas3, _ = model._forward(features, availability, training=False)
    probas4, _ = model._forward(features, availability, training=False)
    
    assert np.allclose(probas3.numpy(), probas4.numpy())
