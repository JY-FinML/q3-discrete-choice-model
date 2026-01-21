"""Plotting utilities for DeepHalo experiments."""

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_training_summary(all_results, output_dir):
    """
    Generate summary plots for all training results.
    
    Creates two plots:
    1. RMSE vs Epoch for all configurations
    2. Final Training RMSE vs Depth
    
    Parameters
    ----------
    all_results : list
        List of result dictionaries from training
    output_dir : str
        Directory to save plots
    """
    print("\nGenerating summary plots...")
    
    # Separate results by parameter set
    param_200k_results = [r for r in all_results if r['config']['param_set'] == 'param_200k']
    param_500k_results = [r for r in all_results if r['config']['param_set'] == 'param_500k']
    
    # Sort by depth
    param_200k_results.sort(key=lambda x: x['config']['depth'])
    param_500k_results.sort(key=lambda x: x['config']['depth'])
    
    # Plot 1: RMSE vs Epoch
    _plot_rmse_vs_epoch(param_200k_results, param_500k_results, output_dir)
    
    # Plot 2: Final Training RMSE vs Depth
    _plot_rmse_vs_depth(param_200k_results, param_500k_results, output_dir)
    
    print("All plots generated successfully!")


def _plot_rmse_vs_epoch(param_200k_results, param_500k_results, output_dir):
    """Plot RMSE vs Epoch for all configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define color maps: blue for 200k, orange-brown for 500k
    blue_cmap = cm.get_cmap('Blues')
    orange_cmap = cm.get_cmap('Oranges')
    
    # Plot param_200k (blue shades)
    if param_200k_results:
        n_configs = len(param_200k_results)
        for i, result in enumerate(param_200k_results):
            depth = result['config']['depth']
            rmse_history = result['training_history']['rmse']
            epochs = range(1, len(rmse_history) + 1)
            
            # Darker blue for larger depth (0.4 to 0.9 range)
            color_intensity = 0.4 + 0.5 * (i / max(1, n_configs - 1))
            color = blue_cmap(color_intensity)
            
            ax.plot(epochs, rmse_history, label=f'200k, d={depth}', 
                    color=color, linewidth=2, alpha=0.8)
    
    # Plot param_500k (orange-brown shades)
    if param_500k_results:
        n_configs = len(param_500k_results)
        for i, result in enumerate(param_500k_results):
            depth = result['config']['depth']
            rmse_history = result['training_history']['rmse']
            epochs = range(1, len(rmse_history) + 1)
            
            # Darker orange for larger depth (0.4 to 0.9 range)
            color_intensity = 0.4 + 0.5 * (i / max(1, n_configs - 1))
            color = orange_cmap(color_intensity)
            
            ax.plot(epochs, rmse_history, label=f'500k, d={depth}', 
                    color=color, linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('RMSE (vs Empirical Frequencies)', fontsize=14)
    ax.set_title('Training RMSE Across All Configurations', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "rmse_vs_epoch_all.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def _plot_rmse_vs_depth(param_200k_results, param_500k_results, output_dir):
    """Plot Final Training RMSE vs Depth."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract final RMSE for each configuration
    if param_200k_results:
        depths_200k = [r['config']['depth'] for r in param_200k_results]
        final_rmse_200k = [r['train_rmse_freq'] for r in param_200k_results]
        ax.plot(depths_200k, final_rmse_200k, 'o-', color='#1f77b4', 
                linewidth=3, markersize=10, label='200k params', alpha=0.8)
        
        # Add value labels
        for depth, rmse in zip(depths_200k, final_rmse_200k):
            ax.annotate(f'{rmse:.4f}', xy=(depth, rmse), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7))
    
    if param_500k_results:
        depths_500k = [r['config']['depth'] for r in param_500k_results]
        final_rmse_500k = [r['train_rmse_freq'] for r in param_500k_results]
        ax.plot(depths_500k, final_rmse_500k, 's-', color='#ff7f0e', 
                linewidth=3, markersize=10, label='500k params', alpha=0.8)
        
        # Add value labels
        for depth, rmse in zip(depths_500k, final_rmse_500k):
            ax.annotate(f'{rmse:.4f}', xy=(depth, rmse), 
                       xytext=(0, -15), textcoords='offset points',
                       ha='center', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', fc='#ffe5cc', alpha=0.7))
    
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Final Training RMSE', fontsize=14)
    ax.set_title('Final Training RMSE vs Model Depth', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    if param_200k_results and param_500k_results:
        all_depths = depths_200k + depths_500k
        ax.set_xticks(range(min(all_depths), max(all_depths) + 1))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "final_rmse_vs_depth.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
