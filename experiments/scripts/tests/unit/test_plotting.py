"""Unit tests for plotting module."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.plotting import plot_training_summary, _plot_rmse_vs_epoch, _plot_rmse_vs_depth


class TestPlottingFunctions:
    """Test cases for plotting functions."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results data for plotting."""
        results = []
        
        # Create param_200k results
        for depth in [3, 4, 5]:
            results.append({
                'config': {
                    'param_set': 'param_200k',
                    'depth': depth,
                    'hidden_dim': 100 + depth * 10,
                },
                'training_history': {
                    'mse': list(np.linspace(0.5, 0.1, 50)),
                    'rmse': list(np.linspace(0.7, 0.3, 50))
                },
                'train_rmse_freq': 0.3 - depth * 0.01
            })
        
        # Create param_500k results
        for depth in [3, 4, 5]:
            results.append({
                'config': {
                    'param_set': 'param_500k',
                    'depth': depth,
                    'hidden_dim': 200 + depth * 10,
                },
                'training_history': {
                    'mse': list(np.linspace(0.4, 0.08, 50)),
                    'rmse': list(np.linspace(0.6, 0.25, 50))
                },
                'train_rmse_freq': 0.25 - depth * 0.01
            })
        
        return results
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for output files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        for f in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    
    def test_plot_training_summary_creates_files(self, sample_results, temp_output_dir):
        """Test that plot_training_summary creates output files."""
        plot_training_summary(sample_results, temp_output_dir)
        
        # Check that plots were created
        expected_files = [
            'rmse_vs_epoch_all.pdf',
            'final_rmse_vs_depth.pdf'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(temp_output_dir, filename)
            assert os.path.exists(filepath), f"Missing file: {filename}"
    
    def test_plot_rmse_vs_epoch_creates_plot(self, sample_results, temp_output_dir):
        """Test that _plot_rmse_vs_epoch creates a valid plot."""
        param_200k = [r for r in sample_results if r['config']['param_set'] == 'param_200k']
        param_500k = [r for r in sample_results if r['config']['param_set'] == 'param_500k']
        
        _plot_rmse_vs_epoch(param_200k, param_500k, temp_output_dir)
        
        output_file = os.path.join(temp_output_dir, 'rmse_vs_epoch_all.pdf')
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    def test_plot_rmse_vs_depth_creates_plot(self, sample_results, temp_output_dir):
        """Test that _plot_rmse_vs_depth creates a valid plot."""
        param_200k = [r for r in sample_results if r['config']['param_set'] == 'param_200k']
        param_500k = [r for r in sample_results if r['config']['param_set'] == 'param_500k']
        
        _plot_rmse_vs_depth(param_200k, param_500k, temp_output_dir)
        
        output_file = os.path.join(temp_output_dir, 'final_rmse_vs_depth.pdf')
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    def test_plot_with_empty_results(self, temp_output_dir):
        """Test plotting handles empty results gracefully."""
        # Should not crash with empty lists
        _plot_rmse_vs_epoch([], [], temp_output_dir)
        _plot_rmse_vs_depth([], [], temp_output_dir)
        
        # Files should still be created (even if empty)
        assert os.path.exists(os.path.join(temp_output_dir, 'rmse_vs_epoch_all.pdf'))
        assert os.path.exists(os.path.join(temp_output_dir, 'final_rmse_vs_depth.pdf'))
    
    def test_plot_with_single_param_set(self, sample_results, temp_output_dir):
        """Test plotting with only one parameter set."""
        param_200k = [r for r in sample_results if r['config']['param_set'] == 'param_200k']
        
        # Should work with only one param set
        _plot_rmse_vs_epoch(param_200k, [], temp_output_dir)
        _plot_rmse_vs_depth(param_200k, [], temp_output_dir)
        
        assert os.path.exists(os.path.join(temp_output_dir, 'rmse_vs_epoch_all.pdf'))
        assert os.path.exists(os.path.join(temp_output_dir, 'final_rmse_vs_depth.pdf'))
    
    def test_plot_sorts_by_depth(self, temp_output_dir):
        """Test that results are sorted by depth."""
        # Create unsorted results
        results = [
            {
                'config': {'param_set': 'param_200k', 'depth': 5, 'hidden_dim': 150},
                'training_history': {'rmse': [0.5, 0.4, 0.3]},
                'train_rmse_freq': 0.3
            },
            {
                'config': {'param_set': 'param_200k', 'depth': 3, 'hidden_dim': 130},
                'training_history': {'rmse': [0.6, 0.5, 0.4]},
                'train_rmse_freq': 0.4
            },
            {
                'config': {'param_set': 'param_200k', 'depth': 4, 'hidden_dim': 140},
                'training_history': {'rmse': [0.55, 0.45, 0.35]},
                'train_rmse_freq': 0.35
            },
        ]
        
        # Should not raise error (sorting happens internally)
        plot_training_summary(results, temp_output_dir)
        
        assert os.path.exists(os.path.join(temp_output_dir, 'rmse_vs_epoch_all.pdf'))
    
    def test_plot_closes_figures(self, sample_results, temp_output_dir):
        """Test that plotting closes matplotlib figures properly."""
        initial_fig_count = len(plt.get_fignums())
        
        plot_training_summary(sample_results, temp_output_dir)
        
        final_fig_count = len(plt.get_fignums())
        
        # No figures should be left open
        assert final_fig_count == initial_fig_count
