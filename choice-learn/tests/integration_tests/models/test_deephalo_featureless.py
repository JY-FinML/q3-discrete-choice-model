"""Integration tests for DeepHaloFeatureless model."""

import numpy as np
import shutil
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models import DeepHaloFeatureless


# Create a larger synthetic dataset for integration testing
np.random.seed(42)
n_samples = 200
n_items = 10
n_features = 1  # Just availability

# Generate random availability masks
availability = np.random.choice([0.0, 1.0], size=(n_samples, n_items), p=[0.2, 0.8])
# Ensure at least 2 items are available per choice
for i in range(n_samples):
    if np.sum(availability[i]) < 2:
        availability[i, :2] = 1.0

# Generate choices based on availability
choices = []
for i in range(n_samples):
    available_indices = np.where(availability[i] == 1.0)[0]
    choices.append(np.random.choice(available_indices))
choices = np.array(choices)

# Reshape for ChoiceDataset
items_features = availability[:, :, np.newaxis]

integration_dataset = ChoiceDataset(
    items_features_by_choice=items_features.astype('float32'),
    available_items_by_choice=availability.astype('float32'),
    choices=choices,
)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_deephalo_run():
    """Dummy test to check that all main methods run without errors."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        opt_size=n_items,
        depth=3,
        resnet_width=32,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=5,
        batch_size=32,
        loss_type='nll'
    )
    
    # Test all main methods
    _ = model.fit(integration_dataset)
    _ = model.evaluate(integration_dataset)
    _ = model.predict_probas(integration_dataset)
    
    assert True


def test_deephalo_fit_nll():
    """Tests that DeepHaloFeatureless can fit with NLL loss."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        opt_size=n_items,
        depth=4,
        resnet_width=32,
        block_types=['qua', 'qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=20,
        batch_size=32,
        loss_type='nll'
    )
    
    nll_before = model.evaluate(integration_dataset)
    history = model.fit(integration_dataset)
    nll_after = model.evaluate(integration_dataset)
    
    # Loss should decrease after training
    assert nll_after < nll_before
    assert 'train_loss' in history
    assert len(history['train_loss']) == 20


def test_deephalo_fit_mse():
    """Tests that DeepHaloFeatureless can fit with MSE loss."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        opt_size=n_items,
        depth=4,
        resnet_width=32,
        block_types=['qua', 'exa', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=20,
        batch_size=32,
        loss_type='mse'
    )
    
    mse_before = model.evaluate(integration_dataset)
    history = model.fit(integration_dataset)
    mse_after = model.evaluate(integration_dataset)
    
    # Loss should decrease after training
    assert mse_after < mse_before
    assert 'train_loss' in history
    assert len(history['train_loss']) == 20


# ============================================================================
# LEARNING CAPABILITY TESTS
# ============================================================================

def test_deephalo_convergence():
    """Tests that model converges on a simple dataset."""
    tf.config.run_functions_eagerly(True)
    
    # Create a simple dataset where first item is always chosen
    simple_availability = np.ones((50, 5), dtype='float32')
    simple_choices = np.zeros(50, dtype=int)  # Always choose item 0
    simple_items_features = simple_availability[:, :, np.newaxis]
    
    simple_dataset = ChoiceDataset(
        items_features_by_choice=simple_items_features,
        available_items_by_choice=simple_availability,
        choices=simple_choices,
    )
    
    model = DeepHaloFeatureless(
        opt_size=5,
        depth=3,
        resnet_width=16,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.01,
        epochs=50,
        batch_size=10,
        loss_type='nll'
    )
    
    history = model.fit(simple_dataset)
    
    # After training, model should predict item 0 with high probability
    probas = model.predict_probas(simple_dataset)
    
    # Item 0 should have highest probability on average
    assert np.mean(probas[:, 0]) > 0.3  # Should be higher than uniform (0.2)
    
    # Training loss should decrease
    assert history['train_loss'][-1] < history['train_loss'][0]


def test_deephalo_predict_accuracy():
    """Tests that predictions improve after training."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        opt_size=n_items,
        depth=4,
        resnet_width=32,
        block_types=['qua', 'exa', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=20,
        batch_size=32,
        loss_type='nll'
    )
    
    # Get predictions before training
    probas_before = model.predict_probas(integration_dataset)
    preds_before = np.argmax(probas_before, axis=-1)
    accuracy_before = np.mean(preds_before == choices)
    
    # Train model
    model.fit(integration_dataset)
    
    # Get predictions after training
    probas_after = model.predict_probas(integration_dataset)
    preds_after = np.argmax(probas_after, axis=-1)
    accuracy_after = np.mean(preds_after == choices)
    
    # Accuracy should improve
    assert accuracy_after > accuracy_before


# ============================================================================
# ADVANCED FEATURES TESTS
# ============================================================================

def test_deephalo_with_validation():
    """Tests training with validation dataset."""
    tf.config.run_functions_eagerly(True)
    
    # Split dataset
    train_size = 150
    train_dataset = ChoiceDataset(
        items_features_by_choice=items_features[:train_size].astype('float32'),
        available_items_by_choice=availability[:train_size].astype('float32'),
        choices=choices[:train_size],
    )
    
    val_dataset = ChoiceDataset(
        items_features_by_choice=items_features[train_size:].astype('float32'),
        available_items_by_choice=availability[train_size:].astype('float32'),
        choices=choices[train_size:],
    )
    
    model = DeepHaloFeatureless(
        opt_size=n_items,
        depth=3,
        resnet_width=32,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=5,
        batch_size=32,
        loss_type='nll'
    )
    
    history = model.fit(train_dataset, val_dataset=val_dataset)
    
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 5
    assert len(history['val_loss']) == 5


def test_deephalo_save_load():
    """Tests model saving and loading."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        opt_size=n_items,
        depth=3,
        resnet_width=32,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=5,
        batch_size=32,
        loss_type='nll'
    )
    
    # Train model
    history = model.fit(integration_dataset)
    
    # Get predictions before saving
    probas_before = model.predict_probas(integration_dataset)
    loss_before = model.evaluate(integration_dataset)
    
    # Save model
    model.save_model("test_deephalo_save")
    
    # Load model
    loaded_model = DeepHaloFeatureless.load_model("test_deephalo_save")
    
    # Get predictions after loading
    probas_after = loaded_model.predict_probas(integration_dataset)
    loss_after = loaded_model.evaluate(integration_dataset)
    
    # Predictions should be identical
    assert np.allclose(probas_before, probas_after, atol=1e-5)
    assert np.isclose(loss_before, loss_after, atol=1e-5)
    
    # Clean up
    shutil.rmtree("test_deephalo_save")


def test_deephalo_batch_prediction_consistency():
    """Tests that predictions are consistent across different batch sizes."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeatureless(
        opt_size=n_items,
        depth=3,
        resnet_width=32,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        loss_type='nll'
    )
    
    # Get predictions with different batch sizes
    probas_batch_1 = model.predict_probas(integration_dataset, batch_size=1)
    probas_batch_10 = model.predict_probas(integration_dataset, batch_size=10)
    probas_batch_all = model.predict_probas(integration_dataset, batch_size=-1)
    
    # All predictions should be identical
    assert np.allclose(probas_batch_1, probas_batch_10, atol=1e-5)
    assert np.allclose(probas_batch_1, probas_batch_all, atol=1e-5)


# ============================================================================
# ARCHITECTURE VARIANTS TESTS
# ============================================================================

def test_deephalo_different_architectures():
    """Tests different architecture configurations."""
    tf.config.run_functions_eagerly(True)
    
    architectures = [
        {'depth': 2, 'block_types': ['qua']},
        {'depth': 3, 'block_types': ['qua', 'qua']},
        {'depth': 3, 'block_types': ['exa', 'exa']},
        {'depth': 4, 'block_types': ['qua', 'exa', 'qua']},
        {'depth': 5, 'block_types': ['qua', 'qua', 'exa', 'exa']},
    ]
    
    for arch in architectures:
        model = DeepHaloFeatureless(
            opt_size=n_items,
            depth=arch['depth'],
            resnet_width=16,
            block_types=arch['block_types'],
            optimizer='Adam',
            lr=0.0001,
            epochs=5,
            batch_size=32,
            loss_type='nll'
        )
        
        history = model.fit(integration_dataset)
        loss = model.evaluate(integration_dataset)
        
        assert 'train_loss' in history
        assert loss > 0.0


# ============================================================================
# LAZY INSTANTIATION TESTS
# ============================================================================

def test_deephalo_lazy_instantiation():
    """Tests that model works without opt_size using lazy instantiation."""
    tf.config.run_functions_eagerly(True)
    
    # Create model without opt_size
    model = DeepHaloFeatureless(
        depth=4,
        resnet_width=32,
        block_types=['qua', 'exa', 'qua'],
        optimizer='Adam',
        lr=0.0001,
        epochs=10,
        batch_size=32,
        loss_type='nll'
    )
    
    assert model.opt_size is None
    assert model.instantiated == False
    
    # Fit should trigger instantiation
    history = model.fit(integration_dataset)
    
    assert model.opt_size == n_items  # Inferred from dataset
    assert model.instantiated == True
    
    # Model should work normally after instantiation
    nll = model.evaluate(integration_dataset)
    probas = model.predict_probas(integration_dataset)
    
    assert nll > 0.0
    assert probas.shape == (n_samples, n_items)
    assert 'train_loss' in history


def test_deephalo_lazy_with_default_blocks():
    """Tests lazy instantiation with default block types."""
    tf.config.run_functions_eagerly(True)
    
    # Create model with minimal parameters
    model = DeepHaloFeatureless(
        depth=3,
        resnet_width=32,
        optimizer='Adam',
        lr=0.0001,
        epochs=10,
        batch_size=32,
        loss_type='nll'
    )
    
    assert model.opt_size is None
    assert model.block_types == ['qua', 'qua']  # Default for depth=3
    assert model.instantiated == False
    
    # Fit should work
    history = model.fit(integration_dataset)
    
    assert model.opt_size == n_items
    assert model.instantiated == True
    assert len(model.blocks) == 2
    assert 'train_loss' in history


def test_deephalo_lazy_instantiation_convergence():
    """Tests that lazily instantiated model can still converge."""
    tf.config.run_functions_eagerly(True)
    
    # Create simple dataset where first item is always chosen
    simple_availability = np.ones((50, 5), dtype='float32')
    simple_choices = np.zeros(50, dtype=int)
    simple_items_features = simple_availability[:, :, np.newaxis]
    
    simple_dataset = ChoiceDataset(
        items_features_by_choice=simple_items_features,
        available_items_by_choice=simple_availability,
        choices=simple_choices,
    )
    
    # Create model without opt_size
    model = DeepHaloFeatureless(
        depth=3,
        resnet_width=16,
        optimizer='Adam',
        lr=0.01,
        epochs=50,
        batch_size=10,
        loss_type='nll'
    )
    
    history = model.fit(simple_dataset)
    
    # After training, model should predict item 0 with high probability
    probas = model.predict_probas(simple_dataset)
    
    assert model.opt_size == 5
    assert np.mean(probas[:, 0]) > 0.3
    assert history['train_loss'][-1] < history['train_loss'][0]
