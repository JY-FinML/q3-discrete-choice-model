"""
Monte Carlo Data Generation for Sparse Discrete Choice Model

This script generates synthetic data for Monte Carlo simulations based on the 
paper "Estimating Discrete Choice Demand Models with Sparse Market-Product Shocks"
by Lu and Shimizu.

The script implements 4 Data Generating Processes (DGPs):
- DGP1: Sparse ξ + Exogenous price
- DGP2: Sparse ξ + Endogenous price  
- DGP3: Non-sparse ξ + Exogenous price
- DGP4: Non-sparse ξ + Endogenous price
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import os
from pathlib import Path


class MonteCarloDataGenerator:
    """Generate synthetic data for Monte Carlo simulations"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator
        
        Parameters:
        -----------
        seed : int
            Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # True parameter values (same for all DGPs)
        self.beta_p_true = -1.0      # Price coefficient mean
        self.beta_w_true = 0.5       # Feature coefficient
        self.sigma_true = 1.5        # Price coefficient std dev
        self.xi_bar_true = -1.0      # Market-level shock
        
        # Cost shock parameters
        self.cost_shock_std = 0.7
        self.cost_coef_w = 0.3       # Coefficient on w in price equation
        
    def simulate_market_shares(
        self, 
        prices: np.ndarray,
        features: np.ndarray, 
        xi: np.ndarray,
        n_sim: int = 1000
    ) -> np.ndarray:
        """
        Simulate market shares using Monte Carlo integration
        
        Parameters:
        -----------
        prices : np.ndarray
            Prices for J products (J,)
        features : np.ndarray  
            Features for J products (J,)
        xi : np.ndarray
            Demand shocks for J products (J,)
        n_sim : int
            Number of Monte Carlo draws
            
        Returns:
        --------
        shares : np.ndarray
            Market shares including outside option (J+1,)
        """
        J = len(prices)
        shares_sum = np.zeros(J + 1)  # +1 for outside option
        
        # Monte Carlo integration over random coefficients
        for i in range(n_sim):
            # Draw consumer i's price coefficient
            beta_pi = self.beta_p_true + self.sigma_true * np.random.randn()
            
            # Compute utilities (deterministic part)
            utilities = np.zeros(J + 1)
            utilities[0] = 0  # Outside option normalized to 0
            
            for j in range(J):
                # delta_jt = beta_bar' X_jt + xi_jt
                delta_jt = self.beta_p_true * prices[j] + self.beta_w_true * features[j] + xi[j]
                
                # mu_ijt = (beta_i - beta_bar)' X_jt
                mu_ijt = (beta_pi - self.beta_p_true) * prices[j]
                
                utilities[j + 1] = delta_jt + mu_ijt
            
            # Logit choice probabilities
            exp_utils = np.exp(utilities)
            probs = exp_utils / np.sum(exp_utils)
            
            shares_sum += probs
        
        # Average over simulations
        shares = shares_sum / n_sim
        
        return shares
    
    def generate_eta_sparse(self, J: int, t: int) -> np.ndarray:
        """
        Generate sparse eta structure (for DGP1 and DGP2)
        
        Parameters:
        -----------
        J : int
            Number of products
        t : int  
            Market index (for reproducibility)
            
        Returns:
        --------
        eta : np.ndarray
            Product-market deviations (J,)
        """
        eta = np.zeros(J)
        
        # First 40% are non-sparse
        cutoff = int(0.4 * J)
        
        for j in range(J):
            if j < cutoff:
                # Non-sparse: alternate between +1 and -1
                eta[j] = 1.0 if j % 2 == 0 else -1.0
            else:
                # Sparse: eta = 0
                eta[j] = 0.0
                
        return eta
    
    def generate_eta_nonsparse(self, J: int, t: int) -> np.ndarray:
        """
        Generate non-sparse eta (for DGP3 and DGP4)
        
        Parameters:
        -----------
        J : int
            Number of products
        t : int
            Market index (for reproducibility)
            
        Returns:
        --------
        eta : np.ndarray
            Product-market deviations (J,)
        """
        # i.i.d. Normal(0, 1/9) - standard deviation = 1/3
        eta = np.random.normal(0, 1/3, size=J)
        return eta
    
    def generate_alpha_exogenous(self, eta: np.ndarray) -> np.ndarray:
        """
        Generate alpha for exogenous price (DGP1 and DGP3)
        
        Parameters:
        -----------
        eta : np.ndarray
            Product-market deviations
            
        Returns:
        --------
        alpha : np.ndarray
            Alpha coefficients (all zeros for exogenous)
        """
        return np.zeros_like(eta)
    
    def generate_alpha_endogenous_sparse(self, eta: np.ndarray) -> np.ndarray:
        """
        Generate alpha for endogenous price with sparse eta (DGP2)
        
        Parameters:
        -----------
        eta : np.ndarray
            Product-market deviations
            
        Returns:
        --------
        alpha : np.ndarray
            Alpha coefficients
        """
        alpha = np.zeros_like(eta)
        
        for j in range(len(eta)):
            if eta[j] == 1.0:
                alpha[j] = 0.3
            elif eta[j] == -1.0:
                alpha[j] = -0.3
            else:  # eta[j] == 0
                alpha[j] = 0.0
                
        return alpha
    
    def generate_alpha_endogenous_nonsparse(self, eta: np.ndarray) -> np.ndarray:
        """
        Generate alpha for endogenous price with non-sparse eta (DGP4)
        
        Parameters:
        -----------
        eta : np.ndarray
            Product-market deviations
            
        Returns:
        --------
        alpha : np.ndarray
            Alpha coefficients
        """
        alpha = np.zeros_like(eta)
        
        for j in range(len(eta)):
            if eta[j] >= 1/3:
                alpha[j] = 0.3
            elif eta[j] <= -1/3:
                alpha[j] = -0.3
            else:
                alpha[j] = 0.0
                
        return alpha
    
    def generate_single_dataset(
        self, 
        T: int, 
        J: int,
        dgp_type: str,
        market_size: int = 1000,
        n_sim: int = 1000,
        replication: int = 0
    ) -> pd.DataFrame:
        """
        Generate a single dataset for one replication
        
        Parameters:
        -----------
        T : int
            Number of markets
        J : int
            Number of products per market
        dgp_type : str
            DGP type: 'dgp1', 'dgp2', 'dgp3', or 'dgp4'
        market_size : int
            Number of potential consumers per market
        n_sim : int
            Number of Monte Carlo draws for share simulation
        replication : int
            Replication number (for seed)
            
        Returns:
        --------
        df : pd.DataFrame
            Dataset with all variables
        """
        # Set seed based on replication
        np.random.seed(self.seed + replication)
        
        data_list = []
        
        for t in range(T):
            # Market-level shock (constant across DGPs)
            xi_bar_t = self.xi_bar_true
            
            # Generate product-level data
            for j in range(J):
                # Product feature w_jt ~ U(1, 2)
                w_jt = np.random.uniform(1, 2)
                
                # Generate eta based on DGP type
                if dgp_type in ['dgp1', 'dgp2']:
                    # Sparse eta
                    eta_all = self.generate_eta_sparse(J, t)
                else:  # dgp3, dgp4
                    # Non-sparse eta
                    eta_all = self.generate_eta_nonsparse(J, t)
                
                eta_jt = eta_all[j]
                
                # Compute xi_jt
                xi_jt = xi_bar_t + eta_jt
                
                # Store for later (need all products in market for share simulation)
                if j == 0:
                    w_t = []
                    xi_t = []
                    eta_t = []
                    
                w_t.append(w_jt)
                xi_t.append(xi_jt)
                eta_t.append(eta_jt)
            
            # Convert to arrays
            w_t = np.array(w_t)
            xi_t = np.array(xi_t)
            eta_t = np.array(eta_t)
            
            # Generate alpha based on DGP type
            if dgp_type == 'dgp1':
                alpha_t = self.generate_alpha_exogenous(eta_t)
            elif dgp_type == 'dgp2':
                alpha_t = self.generate_alpha_endogenous_sparse(eta_t)
            elif dgp_type == 'dgp3':
                alpha_t = self.generate_alpha_exogenous(eta_t)
            else:  # dgp4
                alpha_t = self.generate_alpha_endogenous_nonsparse(eta_t)
            
            # Generate cost shocks and prices
            u_t = np.random.normal(0, self.cost_shock_std, size=J)
            p_t = alpha_t + self.cost_coef_w * w_t + u_t
            
            # Simulate market shares
            shares_t = self.simulate_market_shares(p_t, w_t, xi_t, n_sim=n_sim)
            
            # Convert to quantities
            quantities_t = np.round(market_size * shares_t[1:]).astype(int)  # Exclude outside option
            
            # Store data for this market
            for j in range(J):
                data_list.append({
                    'market': t,
                    'product': j,
                    'price': p_t[j],
                    'feature': w_t[j],
                    'quantity': quantities_t[j],
                    'market_size': market_size,
                    'share': shares_t[j + 1],  # +1 because shares_t includes outside option
                    'outside_share': shares_t[0],
                    'cost_shock': u_t[j],
                    'xi_true': xi_t[j],
                    'eta_true': eta_t[j],
                    'xi_bar_true': xi_bar_t,
                    'alpha_true': alpha_t[j]
                })
        
        df = pd.DataFrame(data_list)
        
        # Add metadata
        df.attrs['dgp_type'] = dgp_type
        df.attrs['T'] = T
        df.attrs['J'] = J
        df.attrs['market_size'] = market_size
        df.attrs['beta_p_true'] = self.beta_p_true
        df.attrs['beta_w_true'] = self.beta_w_true
        df.attrs['sigma_true'] = self.sigma_true
        df.attrs['xi_bar_true'] = self.xi_bar_true
        df.attrs['replication'] = replication
        
        return df
    
    def generate_monte_carlo_data(
        self,
        T_values: List[int] = [25, 100],
        J_values: List[int] = [5, 15],
        dgp_types: List[str] = ['dgp1', 'dgp2', 'dgp3', 'dgp4'],
        n_replications: int = 50,
        market_size: int = 1000,
        output_dir: str = None
    ) -> Dict[str, List[pd.DataFrame]]:
        """
        Generate complete Monte Carlo datasets for all configurations
        
        Parameters:
        -----------
        T_values : List[int]
            List of market counts to test
        J_values : List[int]
            List of product counts to test
        dgp_types : List[str]
            List of DGP types to generate
        n_replications : int
            Number of replications per configuration
        market_size : int
            Number of potential consumers per market
        output_dir : str
            Directory to save datasets (if None, not saved)
            
        Returns:
        --------
        datasets : Dict[str, List[pd.DataFrame]]
            Dictionary mapping config names to lists of datasets
        """
        datasets = {}
        
        total_configs = len(T_values) * len(J_values) * len(dgp_types)
        current_config = 0
        
        for dgp_type in dgp_types:
            for T in T_values:
                for J in J_values:
                    current_config += 1
                    config_name = f"{dgp_type}_T{T}_J{J}"
                    
                    print(f"\nGenerating {config_name} ({current_config}/{total_configs})...")
                    print(f"  DGP: {dgp_type.upper()}, T={T}, J={J}, Replications={n_replications}")
                    
                    config_datasets = []
                    
                    for r in range(n_replications):
                        if r % 10 == 0:
                            print(f"    Replication {r+1}/{n_replications}...", end='\r')
                        
                        df = self.generate_single_dataset(
                            T=T,
                            J=J,
                            dgp_type=dgp_type,
                            market_size=market_size,
                            replication=r
                        )
                        
                        config_datasets.append(df)
                    
                    print(f"    Completed {n_replications} replications   ")
                    
                    datasets[config_name] = config_datasets
                    
                    # Save if output directory specified
                    if output_dir is not None:
                        self.save_config_data(config_datasets, config_name, output_dir)
        
        print(f"\n✓ Generated {total_configs} configurations with {n_replications} replications each")
        print(f"  Total datasets: {total_configs * n_replications}")
        
        return datasets
    
    def save_config_data(
        self, 
        datasets: List[pd.DataFrame], 
        config_name: str,
        output_dir: str
    ):
        """
        Save datasets for a single configuration
        
        Parameters:
        -----------
        datasets : List[pd.DataFrame]
            List of datasets (one per replication)
        config_name : str
            Configuration name
        output_dir : str
            Output directory
        """
        config_dir = Path(output_dir) / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each replication
        for i, df in enumerate(datasets):
            filepath = config_dir / f"replication_{i:03d}.csv"
            df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'dgp_type': datasets[0].attrs['dgp_type'],
            'T': datasets[0].attrs['T'],
            'J': datasets[0].attrs['J'],
            'market_size': datasets[0].attrs['market_size'],
            'beta_p_true': datasets[0].attrs['beta_p_true'],
            'beta_w_true': datasets[0].attrs['beta_w_true'],
            'sigma_true': datasets[0].attrs['sigma_true'],
            'xi_bar_true': datasets[0].attrs['xi_bar_true'],
            'n_replications': len(datasets)
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_csv(config_dir / 'metadata.csv', index=False)
        
        print(f"  ✓ Saved to {config_dir}")


def main():
    """Generate all Monte Carlo datasets"""
    
    # Create generator
    generator = MonteCarloDataGenerator(seed=42)
    
    # Set output directory
    output_dir = Path(__file__).parent
    
    # Generate all configurations
    datasets = generator.generate_monte_carlo_data(
        T_values=[25, 100],
        J_values=[5, 15],
        dgp_types=['dgp1', 'dgp2', 'dgp3', 'dgp4'],
        n_replications=50,
        market_size=1000,
        output_dir=output_dir
    )
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nAll datasets saved to: {output_dir}")
    print("\nGenerated configurations:")
    for config_name in sorted(datasets.keys()):
        print(f"  • {config_name}: {len(datasets[config_name])} replications")


if __name__ == "__main__":
    main()
