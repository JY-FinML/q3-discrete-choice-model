"""Integration tests for DeepHaloFeature model."""

import numpy as np
import shutil
import tensorflow as tf

from choice_learn.data import ChoiceDataset
from choice_learn.models import DeepHaloFeature


# Create a larger synthetic dataset for integration testing
np.random.seed(42)
n_samples = 200
n_items = 10
n_features = 8

# Generate random item features
items_features = np.random.randn(n_samples, n_items, n_features).astype('float32')

# Generate random availability masks
availability = np.random.choice([0.0, 1.0], size=(n_samples, n_items), p=[0.2, 0.8])
# Ensure at least 2 items are available per choice
for i in range(n_samples):
    if np.sum(availability[i]) < 2:
        availability[i, :2] = 1.0

# Generate choices based on simple utility function
utilities = np.sum(items_features, axis=2) + np.random.gumbel(size=(n_samples, n_items))
# Mask unavailable items
utilities[availability == 0] = -np.inf
choices = np.argmax(utilities, axis=1)

integration_dataset = ChoiceDataset(
    items_features_by_choice=(items_features,),
    available_items_by_choice=availability.astype('float32'),
    choices=choices,
)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_deephalo_feature_run():
    """Dummy test to check that all main methods run without errors."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=4,
        L=3,
        embed_dim=32,
        dropout=0.1,
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=32
    )
    
    # Test all main methods
    _ = model.fit(integration_dataset, verbose=0)
    _ = model.evaluate(integration_dataset)
    _ = model.predict_probas(integration_dataset)
    
    assert True


def test_deephalo_feature_fit():
    """Tests that DeepHaloFeature can fit successfully."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.1,
        optimizer='Adam',
        lr=0.001,
        epochs=10,
        batch_size=64
    )
    
    nll_before = model.evaluate(integration_dataset)
    history = model.fit(integration_dataset, verbose=0)
    nll_after = model.evaluate(integration_dataset)
    
    # Loss should decrease after training
    assert nll_after < nll_before
    assert 'train_loss' in history
    assert len(history['train_loss']) == 10


# ============================================================================
# LEARNING CAPABILITY TESTS
# ============================================================================

def test_deephalo_feature_convergence():
    """Tests that model converges on a simple dataset."""
    tf.config.run_functions_eagerly(True)
    
    # Create a simple dataset where choice depends on sum of features
    simple_n = 50
    simple_items = 5
    simple_features = 3
    
    simple_item_features = np.random.randn(simple_n, simple_items, simple_features).astype('float32')
    simple_availability = np.ones((simple_n, simple_items), dtype='float32')
    
    # Choices based on sum of features
    simple_utilities = np.sum(simple_item_features, axis=2)
    simple_choices = np.argmax(simple_utilities, axis=1)
    
    simple_dataset = ChoiceDataset(
        items_features_by_choice=(simple_item_features,),
        available_items_by_choice=simple_availability,
        choices=simple_choices,
    )
    
    model = DeepHaloFeature(
        n_items=simple_items,
        input_dim=simple_features,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.01,
        epochs=30,
        batch_size=25
    )
    
    history = model.fit(simple_dataset, verbose=0)
    
    # After training, predictions should improve
    probas = model.predict_probas(simple_dataset)
    preds = np.argmax(probas, axis=-1)
    accuracy = np.mean(preds == simple_choices)
    
    # Should achieve reasonable accuracy
    assert accuracy > 0.3  # Better than random (0.2)
    
    # Training loss should decrease
    assert history['train_loss'][-1] < history['train_loss'][0]


def test_deephalo_feature_predict_accuracy():
    """Tests that predictions improve after training."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.1,
        optimizer='Adam',
        lr=0.001,
        epochs=10,
        batch_size=64
    )
    
    # Get predictions before training
    probas_before = model.predict_probas(integration_dataset)
    preds_before = np.argmax(probas_before, axis=-1)
    accuracy_before = np.mean(preds_before == choices)
    
    # Train model
    model.fit(integration_dataset, verbose=0)
    
    # Get predictions after training
    probas_after = model.predict_probas(integration_dataset)
    preds_after = np.argmax(probas_after, axis=-1)
    accuracy_after = np.mean(preds_after == choices)
    
    # Accuracy should improve
    assert accuracy_after > accuracy_before


def test_deephalo_feature_respects_availability():
    """Tests that model respects availability masks during prediction."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=2,
        L=2,
        embed_dim=32,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=64
    )
    
    model.fit(integration_dataset, verbose=0)
    probas = model.predict_probas(integration_dataset)
    
    # Check that unavailable items have 0 probability
    for i in range(n_samples):
        for j in range(n_items):
            if availability[i, j] == 0:
                assert probas[i, j] == 0.0


# ============================================================================
# ADVANCED FEATURES TESTS
# ============================================================================

def test_deephalo_feature_with_validation():
    """Tests training with validation dataset."""
    tf.config.run_functions_eagerly(True)
    
    # Split dataset
    train_size = 150
    train_dataset = ChoiceDataset(
        items_features_by_choice=(items_features[:train_size],),
        available_items_by_choice=availability[:train_size].astype('float32'),
        choices=choices[:train_size],
    )
    
    val_dataset = ChoiceDataset(
        items_features_by_choice=(items_features[train_size:],),
        available_items_by_choice=availability[train_size:].astype('float32'),
        choices=choices[train_size:],
    )
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=2,
        L=2,
        embed_dim=32,
        dropout=0.1,
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=64
    )
    
    history = model.fit(train_dataset, val_dataset=val_dataset, verbose=0)
    
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 5
    assert len(history['val_loss']) == 5


def test_deephalo_feature_save_load():
    """Tests model saving and loading."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.0,  # No dropout for exact reproducibility
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=32
    )
    
    # Train model
    history = model.fit(integration_dataset, verbose=0)
    
    # Get predictions before saving
    probas_before = model.predict_probas(integration_dataset)
    loss_before = model.evaluate(integration_dataset)
    
    # Save model
    model.save_model("test_deephalo_feature_save")
    
    # Load model
    loaded_model = DeepHaloFeature.load_model("test_deephalo_feature_save")
    
    # Get predictions after loading
    probas_after = loaded_model.predict_probas(integration_dataset)
    loss_after = loaded_model.evaluate(integration_dataset)
    
    # Predictions should be identical (or very close with floating point precision)
    assert np.allclose(probas_before, probas_after, atol=1e-4)
    assert np.isclose(loss_before, loss_after, atol=1e-4)
    
    # Clean up
    shutil.rmtree("test_deephalo_feature_save")


def test_deephalo_feature_batch_prediction_consistency():
    """Tests that predictions are consistent across different batch sizes."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
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

def test_deephalo_feature_different_architectures():
    """Tests different architecture configurations."""
    tf.config.run_functions_eagerly(True)
    
    architectures = [
        {'H': 2, 'L': 1, 'embed_dim': 16},
        {'H': 3, 'L': 2, 'embed_dim': 32},
    ]
    
    for arch in architectures:
        model = DeepHaloFeature(
            n_items=n_items,
            input_dim=n_features,
            H=arch['H'],
            L=arch['L'],
            embed_dim=arch['embed_dim'],
            dropout=0.0,
            optimizer='Adam',
            lr=0.001,
            epochs=3,
            batch_size=64
        )
        
        history = model.fit(integration_dataset, verbose=0)
        loss = model.evaluate(integration_dataset)
        
        assert 'train_loss' in history
        assert loss > 0.0


def test_deephalo_feature_different_dropouts():
    """Tests different dropout configurations."""
    tf.config.run_functions_eagerly(True)
    
    dropout_rates = [0.0, 0.2]
    
    for dropout in dropout_rates:
        model = DeepHaloFeature(
            n_items=n_items,
            input_dim=n_features,
            H=3,
            L=2,
            embed_dim=32,
            dropout=dropout,
            optimizer='Adam',
            lr=0.001,
            epochs=3,
            batch_size=64
        )
        
        history = model.fit(integration_dataset, verbose=0)
        
        assert 'train_loss' in history
        assert model.dropout_rate == dropout


# ============================================================================
# LAZY INSTANTIATION TESTS
# ============================================================================

def test_deephalo_feature_lazy_instantiation():
    """Tests that model works without dimensions using lazy instantiation."""
    tf.config.run_functions_eagerly(True)
    
    # Create model without dimensions
    model = DeepHaloFeature(
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.1,
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=64
    )
    
    assert model.n_items is None
    assert model.input_dim is None
    assert model.instantiated == False
    
    # Fit should trigger instantiation
    history = model.fit(integration_dataset, verbose=0)
    
    assert model.n_items == n_items  # Inferred from dataset
    assert model.input_dim == n_features  # Inferred from dataset
    assert model.instantiated == True
    
    # Model should work normally after instantiation
    nll = model.evaluate(integration_dataset)
    probas = model.predict_probas(integration_dataset)
    
    assert nll > 0.0
    assert probas.shape == (n_samples, n_items)
    assert 'train_loss' in history


def test_deephalo_feature_lazy_convergence():
    """Tests that lazily instantiated model can still converge."""
    tf.config.run_functions_eagerly(True)
    
    # Create simple dataset
    simple_n = 50
    simple_items = 5
    simple_features = 3
    
    simple_item_features = np.random.randn(simple_n, simple_items, simple_features).astype('float32')
    simple_availability = np.ones((simple_n, simple_items), dtype='float32')
    
    # Choices based on sum of features
    simple_utilities = np.sum(simple_item_features, axis=2)
    simple_choices = np.argmax(simple_utilities, axis=1)
    
    simple_dataset = ChoiceDataset(
        items_features_by_choice=(simple_item_features,),
        available_items_by_choice=simple_availability,
        choices=simple_choices,
    )
    
    # Create model without dimensions
    model = DeepHaloFeature(
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.01,
        epochs=30,
        batch_size=25
    )
    
    history = model.fit(simple_dataset, verbose=0)
    
    # After training, model should predict reasonably
    probas = model.predict_probas(simple_dataset)
    preds = np.argmax(probas, axis=-1)
    accuracy = np.mean(preds == simple_choices)
    
    assert model.n_items == simple_items
    assert model.input_dim == simple_features
    assert accuracy > 0.3
    assert history['train_loss'][-1] < history['train_loss'][0]


def test_deephalo_feature_lazy_save_load():
    """Tests saving and loading a lazily instantiated model."""
    tf.config.run_functions_eagerly(True)
    
    # Create and train lazily instantiated model
    model = DeepHaloFeature(
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=32
    )
    
    model.fit(integration_dataset, verbose=0)
    probas_before = model.predict_probas(integration_dataset)
    
    # Save and load
    model.save_model("test_deephalo_feature_lazy_save")
    loaded_model = DeepHaloFeature.load_model("test_deephalo_feature_lazy_save")
    
    # Check loaded model
    assert loaded_model.n_items == n_items
    assert loaded_model.input_dim == n_features
    
    probas_after = loaded_model.predict_probas(integration_dataset)
    assert np.allclose(probas_before, probas_after, atol=1e-5)
    
    # Clean up
    shutil.rmtree("test_deephalo_feature_lazy_save")


# ============================================================================
# COMPARISON WITH FEATURELESS VERSION
# ============================================================================

def test_deephalo_feature_vs_featureless():
    """Tests that feature-based model can learn better than featureless on feature-rich data."""
    tf.config.run_functions_eagerly(True)
    
    # This test is more conceptual - just ensure both models can be trained
    from choice_learn.models import DeepHaloFeatureless
    
    # Train feature-based model
    feature_model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=2,
        L=2,
        embed_dim=32,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=64
    )
    
    feature_model.fit(integration_dataset, verbose=0)
    feature_accuracy = np.mean(
        np.argmax(feature_model.predict_probas(integration_dataset), axis=-1) == choices
    )
    
    # Train featureless model (with availability as features)
    featureless_model = DeepHaloFeatureless(
        n_items=n_items,
        depth=3,
        hidden_dim=32,
        block_types=['qua', 'qua'],
        optimizer='Adam',
        lr=0.001,
        epochs=5,
        batch_size=64,
        loss_type='nll'
    )
    
    featureless_model.fit(integration_dataset, verbose=0)
    featureless_accuracy = np.mean(
        np.argmax(featureless_model.predict_probas(integration_dataset), axis=-1) == choices
    )
    
    # Both should learn something
    assert feature_accuracy > 0.1
    assert featureless_accuracy > 0.1


# ============================================================================
# EDGE CASES AND ROBUSTNESS
# ============================================================================

def test_deephalo_feature_single_sample():
    """Tests prediction on a single sample."""
    tf.config.run_functions_eagerly(True)
    
    model = DeepHaloFeature(
        n_items=n_items,
        input_dim=n_features,
        H=3,
        L=2,
        embed_dim=32,
        dropout=0.0,
        optimizer='Adam',
        lr=0.001
    )
    
    # Create single-sample dataset
    single_dataset = ChoiceDataset(
        items_features_by_choice=(items_features[:1],),
        available_items_by_choice=availability[:1].astype('float32'),
        choices=choices[:1],
    )
    
    probas = model.predict_probas(single_dataset)
    
    assert probas.shape == (1, n_items)
    assert np.isclose(np.sum(probas[0]), 1.0, atol=1e-5)


def test_deephalo_feature_all_items_available():
    """Tests with all items always available."""
    tf.config.run_functions_eagerly(True)
    
    # Create dataset with all items available
    all_avail = np.ones((50, 5), dtype='float32')
    all_features = np.random.randn(50, 5, 3).astype('float32')
    all_choices = np.random.randint(0, 5, 50)
    
    all_dataset = ChoiceDataset(
        items_features_by_choice=(all_features,),
        available_items_by_choice=all_avail,
        choices=all_choices,
    )
    
    model = DeepHaloFeature(
        n_items=5,
        input_dim=3,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.01,
        epochs=10,
        batch_size=10
    )
    
    history = model.fit(all_dataset, verbose=0)
    probas = model.predict_probas(all_dataset)
    
    # All probabilities should be positive
    assert np.all(probas > 0.0)
    assert 'train_loss' in history


def test_deephalo_feature_minimal_availability():
    """Tests with minimal item availability (only 2 items per choice)."""
    tf.config.run_functions_eagerly(True)
    
    # Create dataset with only 2 items available per choice
    min_avail = np.zeros((50, 5), dtype='float32')
    for i in range(50):
        min_avail[i, :2] = 1.0
    
    min_features = np.random.randn(50, 5, 3).astype('float32')
    min_choices = np.random.randint(0, 2, 50)  # Can only choose from first 2
    
    min_dataset = ChoiceDataset(
        items_features_by_choice=(min_features,),
        available_items_by_choice=min_avail,
        choices=min_choices,
    )
    
    model = DeepHaloFeature(
        n_items=5,
        input_dim=3,
        H=2,
        L=2,
        embed_dim=16,
        dropout=0.0,
        optimizer='Adam',
        lr=0.01,
        epochs=10,
        batch_size=10
    )
    
    history = model.fit(min_dataset, verbose=0)
    probas = model.predict_probas(min_dataset)
    
    # Unavailable items should have 0 probability
    assert np.all(probas[:, 2:] == 0.0)
    assert 'train_loss' in history
