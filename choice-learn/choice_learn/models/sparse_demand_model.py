"""Sparse Discrete Choice Demand Model

This module implements "Estimating Discrete Choice Demand Models with Sparse Market-Product Shocks" by Lu and Shimizu.
The model uses Bayesian shrinkage with spike-and-slab priors to estimate random coefficient logit demand with sparse market-product shocks.

Key Features:
- Random coefficient logit with heterogeneous price sensitivity
- Sparse structure: ξ_jt = ξ̄_t + η_jt where many η_jt = 0
- Spike-and-slab prior for sparsity
- TensorFlow Probability for Bayesian inference
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .base_model import ChoiceModel

tfd = tfp.distributions


class SparseDiscreteChoiceModel(ChoiceModel):
    """Sparse Discrete Choice Demand Model.
    
    Implements random coefficient logit with sparse market-product shocks
    using Bayesian shrinkage estimation.
    
    Model Specification:
    --------------------
    u_ijt = X_jt' β_i + ξ_jt + ε_ijt
    
    where:
    - β_i ~ N(β̄, Σ): Random coefficients (heterogeneous preferences)
    - ξ_jt = ξ̄_t + η_jt: Demand shock (market-level + product-market deviation)
    - η_jt has spike-and-slab prior for sparsity
    - ε_ijt ~ Gumbel(0,1): i.i.d. logit errors
    
    Parameters
    ----------
    n_features : int
        Number of product features (including price)
    n_markets : int, optional
        Number of markets. If None, inferred from data
    n_products_per_market : int, optional
        Number of products per market. If None, inferred from data
    n_simulations : int, optional
        Number of Monte Carlo draws for integration, by default 200
    tau_0_sq : float, optional
        Spike variance (for sparse η), by default 0.001
    tau_1_sq : float, optional
        Slab variance (for non-sparse η), by default 1.0
    optimizer : str, optional
        Optimizer type (not used for MCMC), by default 'Adam'
    **kwargs
        Additional arguments passed to ChoiceModel base class
        
    Attributes
    ----------
    beta_mean : tf.Variable
        Mean of random coefficients (n_features,)
    log_sigma : tf.Variable
        Log standard deviations of random coefficients (n_features,)
    xi_bar : tf.Variable
        Market-level shocks (n_markets,)
    eta : tf.Variable
        Product-market deviations (n_markets, n_products_per_market)
    gamma : tf.Variable
        Sparsity indicators (n_markets, n_products_per_market)
    """
    
    def __init__(
        self,
        n_features=2,
        n_markets=None,
        n_products_per_market=None,
        n_simulations=200,
        tau_0_sq=0.001,
        tau_1_sq=1.0,
        optimizer='mcmc',
        n_iterations=10000,
        n_burnin=3000,
        mcmc_step_sizes=None,
        **kwargs
    ):
        """Initialize Sparse Discrete Choice Model.
        
        Parameters
        ----------
        optimizer : str
            'mcmc' for MCMC sampling (default), 'adam'/'sgd' for gradient descent
        n_iterations : int
            Total MCMC iterations (only for mcmc optimizer)
        n_burnin : int
            Burn-in period (only for mcmc optimizer)
        mcmc_step_sizes : dict, optional
            Step sizes for RWMH samplers, e.g. {'log_sigma': 0.1, 'xi_bar': 0.5}
        """
        super().__init__(optimizer=optimizer if optimizer != 'mcmc' else 'adam', **kwargs)
        
        self.n_features = n_features
        self.n_markets = n_markets
        self.n_products_per_market = n_products_per_market
        self.n_simulations = n_simulations
        self.tau_0_sq = tau_0_sq
        self.tau_1_sq = tau_1_sq
        
        # MCMC settings
        self.estimation_method = optimizer.lower()
        self.n_iterations = n_iterations
        self.n_burnin = n_burnin
        self.mcmc_step_sizes = mcmc_step_sizes or {
            'log_sigma': 0.1,
            'xi_bar': 0.5
        }
        
        self.instantiated = False
        
        # Initialize parameters if dimensions are known
        if n_markets is not None and n_products_per_market is not None:
            self._initialize_parameters()
            self.instantiated = True
    
    def instantiate(self, choice_dataset):
        """Instantiate model from dataset.
        
        NOTE: This method is only for compatibility with ChoiceModel base class.
        For MCMC estimation, initialization happens in _fit_mcmc() instead.
        
        Parameters
        ----------
        choice_dataset : ChoiceDataset
            Dataset to infer dimensions from
            
        Returns
        -------
        tuple
            (indexes, weights) for compatibility with base class
        """
        if not self.instantiated:
            # Minimal initialization for base class compatibility
            if self.n_markets is None:
                self.n_markets = 1  # Placeholder
            if self.n_products_per_market is None:
                n_items = choice_dataset.get_n_items() if hasattr(choice_dataset, 'get_n_items') else 5
                self.n_products_per_market = n_items
            
            self._initialize_parameters()
            self.instantiated = True
        
        return {}, self.trainable_weights
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        # β̄: Mean of random coefficients
        self.beta_mean = tf.Variable(
            tf.zeros([self.n_features]),
            dtype=tf.float32,
            name='beta_mean'
        )
        
        # log(σ): Log standard deviations of random coefficients
        self.log_sigma = tf.Variable(
            tf.zeros([self.n_features]),
            dtype=tf.float32,
            name='log_sigma'
        )
        
        # ξ̄_t: Market-level shocks
        self.xi_bar = tf.Variable(
            tf.zeros([self.n_markets]),
            dtype=tf.float32,
            name='xi_bar'
        )
        
        # η_jt: Product-market deviations (sparse)
        self.eta = tf.Variable(
            tf.zeros([self.n_markets, self.n_products_per_market]),
            dtype=tf.float32,
            name='eta'
        )
        
        # γ_jt: Sparsity indicators (0 or 1)
        # Initialized as continuous [0,1] for optimization, will threshold for inference
        self.gamma = tf.Variable(
            0.1 * tf.ones([self.n_markets, self.n_products_per_market]),
            dtype=tf.float32,
            name='gamma'
        )
        
        # φ_t: Inclusion probability for each market
        self.phi = tf.Variable(
            0.5 * tf.ones([self.n_markets]),
            dtype=tf.float32,
            name='phi'
        )
        
        # Pre-generate Monte Carlo draws (fixed for efficiency)
        self.v_draws = tf.constant(
            np.random.randn(self.n_simulations, self.n_features),
            dtype=tf.float32,
            name='v_draws'
        )
    
    @property
    def trainable_weights(self):
        """Return list of trainable parameters."""
        if not self.instantiated:
            return []
        return [self.beta_mean, self.log_sigma, self.xi_bar, self.eta, self.gamma, self.phi]
    
    def compute_market_shares(self, X, market_indices, product_indices):
        """Compute market shares using Monte Carlo integration.
        
        CORRECTED: Now properly normalizes within markets including outside option.
        
        Parameters
        ----------
        X : tf.Tensor
            Product features of shape (n_obs, n_features)
            First feature should be price
        market_indices : tf.Tensor
            Market index for each observation (n_obs,)
        product_indices : tf.Tensor
            Product index within market for each observation (n_obs,)
            
        Returns
        -------
        tf.Tensor
            Market shares (n_obs,) - properly normalized within markets
        """
        # Get xi_jt = xi_bar_t + eta_jt
        xi_bar_t = tf.gather(self.xi_bar, market_indices)  # (n_obs,)
        eta_jt = tf.gather_nd(
            self.eta,
            tf.stack([market_indices, product_indices], axis=1)
        )  # (n_obs,)
        xi_jt = xi_bar_t + eta_jt  # (n_obs,)
        
        # Compute σ = exp(log_sigma)
        sigma = tf.exp(self.log_sigma)  # (n_features,)
        
        # Mean utility: δ_jt = X_jt' β̄ + ξ_jt
        delta_jt = tf.reduce_sum(X * self.beta_mean, axis=1) + xi_jt  # (n_obs,)
        
        # Monte Carlo integration
        # (n_sims, n_features) * (n_features,) = (n_sims, n_features)
        beta_deviations = self.v_draws * sigma[None, :]  
        
        # (n_sims, n_features) @ (n_features, n_obs) = (n_sims, n_obs)
        mu_ij = tf.matmul(beta_deviations, tf.transpose(X))
        
        # Total utility: u_ijt = δ_jt + μ_ijt
        utilities = delta_jt[None, :] + mu_ij  # (n_sims, n_obs)
        
        # CORRECTION: Proper market-level normalization with outside option
        # Group utilities by market and normalize
        exp_utilities = tf.exp(utilities)  # (n_sims, n_obs)
        
        # For each market, sum exp(utilities) of all products in that market
        # Then compute share = exp(u_ijt) / (1 + sum_k_in_market exp(u_ikt))
        
        # VECTORIZED: Avoid Python for loop for speed
        # Reshape to (n_sims * n_obs,) for vectorized segment_sum
        n_obs = tf.shape(X)[0]
        n_markets = tf.reduce_max(market_indices) + 1
        
        # Flatten exp_utilities and replicate market_indices for each simulation
        exp_utilities_flat = tf.reshape(exp_utilities, [-1])  # (n_sims * n_obs,)
        
        # Create market indices for flattened array: repeat market_indices n_sims times
        market_indices_tiled = tf.tile(market_indices[None, :], [self.n_simulations, 1])
        market_indices_flat = tf.reshape(market_indices_tiled, [-1])  # (n_sims * n_obs,)
        
        # Create segment IDs: market_idx + sim_idx * n_markets
        sim_indices = tf.repeat(tf.range(self.n_simulations), n_obs)  # (n_sims * n_obs,)
        segment_ids = market_indices_flat + sim_indices * n_markets  # (n_sims * n_obs,)
        
        # Sum by (sim, market) segments
        market_sums_flat = tf.math.unsorted_segment_sum(
            exp_utilities_flat,
            segment_ids,
            self.n_simulations * n_markets
        )  # (n_sims * n_markets,)
        
        # Reshape back to (n_sims, n_markets)
        market_sums = tf.reshape(market_sums_flat, [self.n_simulations, n_markets])
        
        # Gather market sums for each observation: (n_sims, n_obs)
        obs_market_sums = tf.gather(market_sums, market_indices, axis=1)
        
        # Logit probabilities with outside option (utility = 0)
        # P(j) = exp(u_j) / (1 + sum_k exp(u_k))
        probabilities = exp_utilities / (1.0 + obs_market_sums)  # (n_sims, n_obs)
        
        # Average over simulations
        shares = tf.reduce_mean(probabilities, axis=0)  # (n_obs,)
        
        return shares
    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices
    ):
        """Compute utilities for a batch.
        
        NOTE: This method is only for compatibility with ChoiceModel base class.
        For MCMC estimation, this is NOT used. Use compute_market_shares instead.
        
        Parameters
        ----------
        shared_features_by_choice : unused
        items_features_by_choice : tf.Tensor
            Product features (n_choices, n_items, n_features)
        available_items_by_choice : unused
        choices : unused
            
        Returns
        -------
        tf.Tensor
            Utilities (n_choices, n_items)
        """
        # Simplified stub - mean utilities only (no random coefficients)
        n_choices = tf.shape(items_features_by_choice)[0]
        n_items = tf.shape(items_features_by_choice)[1]
        
        # Basic utility: X' β̄ (no market-product shocks)
        # Shape: (n_choices, n_items, n_features) @ (n_features,) -> (n_choices, n_items)
        utilities = tf.reduce_sum(
            items_features_by_choice * self.beta_mean[None, None, :],
            axis=2
        )
        
        return utilities
    
    def log_prior(self):
        """Compute log prior probability.
        
        Returns
        -------
        tf.Tensor
            Log prior probability (scalar)
        """
        log_prob = 0.0
        
        # Prior for β̄: N(0, V_β) where V_β = 10*I (VARIANCE)
        # tfd.Normal(loc, scale) where scale is STANDARD DEVIATION (not variance)
        # Therefore: scale = sqrt(variance) = sqrt(10)
        log_prob += tf.reduce_sum(
            tfd.Normal(0.0, tf.sqrt(10.0)).log_prob(self.beta_mean)
        )
        
        # Prior for log(σ): r_k ~ N(0, V_r) where V_r = 0.5 (VARIANCE)
        # Paper: "For the log standard deviations, we let V_{r,k}=0.5"
        # tfd.Normal(loc, scale) where scale is STANDARD DEVIATION
        # Therefore: scale = sqrt(variance) = sqrt(0.5) ≈ 0.707
        log_prob += tf.reduce_sum(
            tfd.Normal(0.0, tf.sqrt(0.5)).log_prob(self.log_sigma)
        )
        
        # Prior for ξ̄_t: N(0, V_ξ) where V_ξ = 10 (VARIANCE)
        # tfd.Normal(loc, scale) where scale is STANDARD DEVIATION
        # Therefore: scale = sqrt(variance) = sqrt(10)
        log_prob += tf.reduce_sum(
            tfd.Normal(0.0, tf.sqrt(10.0)).log_prob(self.xi_bar)
        )
        
        # Spike-and-slab prior for η_jt
        # η_jt | γ_jt ~ (1-γ_jt) N(0, τ_0²) + γ_jt N(0, τ_1²)
        # where τ_0² and τ_1² are VARIANCES
        
        # For continuous relaxation during optimization:
        # Use mixture with gamma as mixture weight
        # tfd.Normal(loc, scale) where scale = sqrt(variance)
        spike_log_prob = tfd.Normal(0.0, tf.sqrt(self.tau_0_sq)).log_prob(self.eta)  # scale = sqrt(0.001)
        slab_log_prob = tfd.Normal(0.0, tf.sqrt(self.tau_1_sq)).log_prob(self.eta)   # scale = sqrt(1.0)
        
        # Mixture log probability
        mixture_log_prob = tf.math.log(
            (1 - self.gamma) * tf.exp(spike_log_prob) +     # τ₀² = 0.001
            self.gamma * tf.exp(slab_log_prob) +            # τ₁² = 1.0
            1e-10  # Numerical stability
        )
        log_prob += tf.reduce_sum(mixture_log_prob)
        
        # Prior for φ_t: Beta(1, 1) = Uniform(0, 1)
        # Clip phi to valid range [eps, 1-eps] for numerical stability
        phi_clipped = tf.clip_by_value(self.phi, 1e-6, 1.0 - 1e-6)
        log_prob += tf.reduce_sum(
            tfd.Beta(1.0, 1.0).log_prob(phi_clipped)
        )
        
        # Prior for γ_jt: Bernoulli(φ_t)
        # For continuous relaxation, use Bernoulli log_prob
        # gamma should be in [0, 1], clip for safety
        gamma_clipped = tf.clip_by_value(self.gamma, 1e-6, 1.0 - 1e-6)
        
        # VECTORIZED: Avoid Python for loop - expand phi_t to match gamma shape
        phi_expanded = tf.expand_dims(phi_clipped, axis=1)  # (n_markets, 1)
        log_prob += tf.reduce_sum(
            gamma_clipped * tf.math.log(phi_expanded) +
            (1.0 - gamma_clipped) * tf.math.log(1.0 - phi_expanded)
        )
        
        return log_prob
    
    def log_likelihood(self, X, quantities, market_indices, product_indices, market_sizes):
        """Compute log likelihood from quantities.
        
        L(θ | q) = Π_t Π_j σ_jt^{q_jt}
        log L = Σ_t Σ_j q_jt * log(σ_jt)
        
        Parameters
        ----------
        X : tf.Tensor
            Product features (n_obs, n_features)
        quantities : tf.Tensor
            Observed quantities (n_obs,)
        market_indices : tf.Tensor
            Market indices (n_obs,)
        product_indices : tf.Tensor
            Product indices (n_obs,)
        market_sizes : tf.Tensor
            Market sizes (n_markets,) or (n_obs,)
            
        Returns
        -------
        tf.Tensor
            Log likelihood (scalar)
        """
        # Compute predicted market shares
        shares = self.compute_market_shares(X, market_indices, product_indices)
        
        # Compute log likelihood: Σ q_jt * log(σ_jt)
        # Add small epsilon for numerical stability
        log_shares = tf.math.log(shares + 1e-10)
        log_lik = tf.reduce_sum(tf.cast(quantities, tf.float32) * log_shares)
        
        return log_lik
    
    def log_posterior(self, X, quantities, market_indices, product_indices, market_sizes):
        """Compute log posterior probability.
        
        Parameters
        ----------
        X : tf.Tensor
            Product features
        quantities : tf.Tensor
            Observed quantities
        market_indices : tf.Tensor
            Market indices
        product_indices : tf.Tensor
            Product indices
        market_sizes : tf.Tensor
            Market sizes
            
        Returns
        -------
        tf.Tensor
            Log posterior probability (scalar)
        """
        log_lik = self.log_likelihood(X, quantities, market_indices, product_indices, market_sizes)
        log_prior = self.log_prior()
        return log_lik + log_prior
    
    def fit(self, choice_dataset, verbose=1, return_samples=False):
        """Fit the model using MCMC or gradient descent.
        
        Parameters
        ----------
        choice_dataset : ChoiceDataset or dict
            Training data. Can be ChoiceDataset or dict with keys:
            'X', 'quantities', 'market_indices', 'product_indices', 'market_sizes'
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=detailed)
        return_samples : bool
            If True, return full MCMC samples in addition to summaries
            
        Returns
        -------
        dict
            Posterior summaries (for MCMC) or training history (for gradient descent)
        """
        if self.estimation_method == 'mcmc':
            return self._fit_mcmc(choice_dataset, verbose, return_samples)
        else:
            # Use standard gradient descent from base class
            return super().fit(choice_dataset, verbose=verbose)
    
    def _fit_mcmc(self, data, verbose=1, return_samples=False):
        """Fit model using MCMC sampling.
        
        Parameters
        ----------
        data : dict
            Must contain: X, quantities, market_indices, product_indices, market_sizes
        verbose : int
            Verbosity level
        return_samples : bool
            Whether to return full posterior samples
            
        Returns
        -------
        dict
            Posterior summaries with keys: beta_mean, sigma, xi_bar, eta, gamma, phi
        """
        from scipy.optimize import minimize
        from scipy.stats import beta as beta_dist
        
        # Extract data
        if isinstance(data, dict):
            X = tf.constant(data['X'], dtype=tf.float32)
            quantities = tf.constant(data['quantities'], dtype=tf.float32)
            market_indices = tf.constant(data['market_indices'], dtype=tf.int32)
            product_indices = tf.constant(data['product_indices'], dtype=tf.int32)
            market_sizes = tf.constant(data['market_sizes'], dtype=tf.float32)
        else:
            raise ValueError("For MCMC, data must be a dict with X, quantities, market_indices, product_indices, market_sizes")
        
        # Initialize if needed
        if not self.instantiated:
            n_markets = int(np.max(market_indices.numpy())) + 1
            n_products = int(np.max(product_indices.numpy())) + 1
            self.n_markets = n_markets
            self.n_products_per_market = n_products
            self._initialize_parameters()
            self.instantiated = True
        
        # MCMC sampling
        samples = {
            'beta_mean': [],
            'log_sigma': [],
            'xi_bar': [],
            'eta': [],
            'gamma': [],
            'phi': []
        }
        
        accept_counts = {
            'beta_mean': 0,
            'log_sigma': 0,
            'xi_bar': 0,
            'eta': 0
        }
        
        if verbose > 0:
            print(f"\n{'='*70}")
            print(f"MCMC Estimation: {self.n_iterations} iterations ({self.n_burnin} burn-in)")
            print(f"{'='*70}\n")
            # Determine progress reporting frequency
            if self.n_iterations <= 100:
                report_freq = 10
            elif self.n_iterations <= 500:
                report_freq = 50
            elif self.n_iterations <= 2000:
                report_freq = 100
            else:
                report_freq = 500
        
        # MCMC loop
        for iteration in range(self.n_iterations):
            # Step 1: Update β̄ using TMH
            accepted = self._mcmc_update_beta_mean(X, quantities, market_indices, 
                                                   product_indices, market_sizes)
            accept_counts['beta_mean'] += accepted
            
            # Step 2: Update log σ using RWMH  
            accepted = self._mcmc_update_log_sigma(X, quantities, market_indices,
                                                   product_indices, market_sizes)
            accept_counts['log_sigma'] += accepted
            
            # Step 3: Update ξ̄_t using RWMH
            accepted = self._mcmc_update_xi_bar(X, quantities, market_indices,
                                               product_indices, market_sizes)
            accept_counts['xi_bar'] += accepted
            
            # Step 4: Update η_t using TMH
            accepted = self._mcmc_update_eta(X, quantities, market_indices,
                                            product_indices, market_sizes)
            accept_counts['eta'] += accepted
            
            # Step 5: Update γ_jt using Gibbs
            self._mcmc_update_gamma()
            
            # Step 6: Update φ_t using conjugate Beta
            self._mcmc_update_phi()
            
            # Store samples (after burn-in)
            if iteration >= self.n_burnin:
                samples['beta_mean'].append(self.beta_mean.numpy().copy())
                samples['log_sigma'].append(self.log_sigma.numpy().copy())
                samples['xi_bar'].append(self.xi_bar.numpy().copy())
                samples['eta'].append(self.eta.numpy().copy())
                samples['gamma'].append(self.gamma.numpy().copy())
                samples['phi'].append(self.phi.numpy().copy())
            
            # Progress reporting (adaptive frequency)
            if verbose > 0:
                if iteration == 0:
                    # Always report first iteration
                    print(f"Iteration {iteration + 1}/{self.n_iterations} (initializing...)")
                elif (iteration + 1) % report_freq == 0:
                    log_post = self.log_posterior(X, quantities, market_indices,
                                                  product_indices, market_sizes).numpy()
                    print(f"Iteration {iteration + 1}/{self.n_iterations}, "
                          f"Log-posterior: {log_post:.2f}")
                
                if verbose > 1 and (iteration + 1) % report_freq == 0:
                    for key in accept_counts:
                        rate = accept_counts[key] / (iteration + 1)
                        print(f"  {key} acceptance: {rate:.3f}")
        
        # Compute summaries
        summaries = self._compute_mcmc_summaries(samples, accept_counts)
        
        if verbose > 0:
            self._print_mcmc_results(summaries)
        
        # Mark as fitted
        self.is_fitted = True
        
        if return_samples:
            return {'summaries': summaries, 'samples': samples}
        return summaries
    
    def _find_mode_newton(self, param_name, X, quantities, market_indices, product_indices, market_sizes, max_iter=10):
        """Find mode using simplified Newton's method.
        
        NOTE: This is a SIMPLIFIED implementation using gradient-based approximations.
        For production use, should implement true Hessian computation.
        Paper recommends using full Newton's method with exact Hessian.
        
        Parameters
        ----------
        param_name : str
            'beta_mean', 'eta', etc.
        max_iter : int
            Maximum Newton iterations
            
        Returns
        -------
        tuple
            (mode, hessian_inv) where mode is θ̂ and hessian_inv is V̂_θ (approximated)
        """
        # Start from current value
        if param_name == 'beta_mean':
            current = self.beta_mean.numpy()
        elif param_name == 'eta':
            current = self.eta.numpy()
        else:
            return None, None
        
        # Newton's method: θ^{k+1} = θ^k - H^{-1} g
        for _ in range(max_iter):
            # Compute gradient and Hessian using TensorFlow autodiff
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.trainable_weights)
                log_post = self.log_posterior(X, quantities, market_indices, 
                                             product_indices, market_sizes)
            
            # Get gradient
            if param_name == 'beta_mean':
                grad = tape.gradient(log_post, self.beta_mean).numpy()
            elif param_name == 'eta':
                grad = tape.gradient(log_post, self.eta).numpy()
            
            # Compute Hessian using second-order gradients
            # For efficiency, use approximation: H ≈ -g g^T (outer product approximation)
            # In full implementation, should compute true Hessian
            # Here we use empirical Fisher information as approximation
            hessian_approx = -np.outer(grad.flatten(), grad.flatten()) / (np.linalg.norm(grad) + 1e-6)
            
            # Check convergence
            if np.linalg.norm(grad) < 1e-4:
                break
            
            # Newton step (simplified with regularization)
            try:
                step = np.linalg.solve(hessian_approx + 1e-4 * np.eye(len(grad.flatten())), 
                                      grad.flatten())
                current_flat = current.flatten() - 0.5 * step  # Damped Newton
                current = current_flat.reshape(current.shape)
            except:
                # If Hessian is singular, use gradient ascent
                current = current + 0.01 * grad
            
            # Update parameter
            if param_name == 'beta_mean':
                self.beta_mean.assign(current.astype(np.float32))
            elif param_name == 'eta':
                self.eta.assign(current.astype(np.float32))
            
            del tape
        
        # Compute Hessian at mode (negative for variance)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_weights)
            log_post = self.log_posterior(X, quantities, market_indices,
                                         product_indices, market_sizes)
        
        if param_name == 'beta_mean':
            grad = tape.gradient(log_post, self.beta_mean).numpy()
        elif param_name == 'eta':
            grad = tape.gradient(log_post, self.eta).numpy()
        
        # Approximation: V̂ = (g g^T)^{-1} (outer product based)
        # In production, should use true Hessian
        hessian_inv = np.eye(len(grad.flatten())) * (1.0 / (np.var(grad) + 1e-4))
        
        del tape
        return current, hessian_inv
    
    def _mcmc_update_beta_mean(self, X, quantities, market_indices, product_indices, market_sizes):
        """Update β̄ using Tailored Metropolis-Hastings.
        
        Implements paper's TMH algorithm:
        1. Find mode θ̂ using Newton's method
        2. Compute Hessian V̂_θ at mode
        3. Propose from N(θ̂, κ²_θ V̂_θ) where κ_θ = 2.38/√dim(θ)
        4. Accept with TMH probability (includes proposal density)
        """
        beta_current = self.beta_mean.numpy()
        dim_beta = len(beta_current)
        
        # Step 1 & 2: Find mode and Hessian
        beta_mode, V_beta = self._find_mode_newton('beta_mean', X, quantities, 
                                                    market_indices, product_indices, market_sizes)
        
        # Restore current value
        self.beta_mean.assign(beta_current.astype(np.float32))
        
        # Step 3: Proposal from N(θ̂, κ²_θ V̂_θ)
        kappa = 2.38 / np.sqrt(dim_beta)  # Optimal scaling from Gelman et al.
        proposal_cov = kappa**2 * V_beta
        
        try:
            beta_proposal = np.random.multivariate_normal(
                beta_mode.flatten(), 
                proposal_cov
            ).reshape(beta_current.shape)
        except:
            # Fallback if covariance is singular
            beta_proposal = beta_mode + kappa * np.random.randn(*beta_current.shape) * np.sqrt(np.diag(V_beta).mean())
        
        # Step 4: Compute acceptance probability with proposal densities
        # log α = log π(θ̃|•) + log q(θ^{(g)}|θ̂) - log π(θ^{(g)}|•) - log q(θ̃|θ̂)
        
        # Posterior at proposal
        self.beta_mean.assign(beta_proposal.astype(np.float32))
        log_post_proposal = self.log_posterior(X, quantities, market_indices,
                                               product_indices, market_sizes).numpy()
        
        # Posterior at current
        self.beta_mean.assign(beta_current.astype(np.float32))
        log_post_current = self.log_posterior(X, quantities, market_indices,
                                              product_indices, market_sizes).numpy()
        
        # Proposal densities (multivariate normal)
        try:
            from scipy.stats import multivariate_normal
            log_q_current_given_mode = multivariate_normal.logpdf(
                beta_current.flatten(), mean=beta_mode.flatten(), cov=proposal_cov
            )
            log_q_proposal_given_mode = multivariate_normal.logpdf(
                beta_proposal.flatten(), mean=beta_mode.flatten(), cov=proposal_cov
            )
        except:
            # Symmetric proposal fallback
            log_q_current_given_mode = 0
            log_q_proposal_given_mode = 0
        
        # TMH acceptance probability
        log_accept = (log_post_proposal + log_q_current_given_mode - 
                     log_post_current - log_q_proposal_given_mode)
        
        # Accept/reject
        if np.log(np.random.rand()) < log_accept:
            self.beta_mean.assign(beta_proposal.astype(np.float32))
            return 1
        return 0
    
    def _mcmc_update_log_sigma(self, X, quantities, market_indices, product_indices, market_sizes):
        """Update log σ using Random Walk MH with adaptive covariance.
        
        Uses proposal: r̃ ~ N(r^{(g)}, κ_r S_r)
        where S_r can be adapted during burn-in.
        """
        log_sigma_current = self.log_sigma.numpy()
        dim_r = len(log_sigma_current)
        
        # Use step_size as scaling factor κ_r
        kappa_r = self.mcmc_step_sizes.get('log_sigma', 0.1)
        
        # For simplicity, use diagonal covariance (can be improved with full covariance)
        # In full implementation, S_r should be estimated from samples during burn-in
        S_r = np.eye(dim_r)
        
        # Proposal: r̃ ~ N(r^{(g)}, κ_r² S_r)
        log_sigma_proposal = log_sigma_current + kappa_r * np.random.multivariate_normal(
            np.zeros(dim_r), S_r
        )
        
        self.log_sigma.assign(log_sigma_proposal.astype(np.float32))
        log_post_proposal = self.log_posterior(X, quantities, market_indices,
                                               product_indices, market_sizes).numpy()
        
        self.log_sigma.assign(log_sigma_current.astype(np.float32))
        log_post_current = self.log_posterior(X, quantities, market_indices,
                                              product_indices, market_sizes).numpy()
        
        # RWMH acceptance (symmetric proposal)
        log_accept = log_post_proposal - log_post_current
        if np.log(np.random.rand()) < log_accept:
            self.log_sigma.assign(log_sigma_proposal.astype(np.float32))
            return 1
        return 0
    
    def _mcmc_update_xi_bar(self, X, quantities, market_indices, product_indices, market_sizes):
        """Update ξ̄_t using RWMH for each market."""
        xi_bar_current = self.xi_bar.numpy()
        step_size = self.mcmc_step_sizes['xi_bar']
        n_accepts = 0
        
        for t in range(len(xi_bar_current)):
            xi_bar_proposal = xi_bar_current.copy()
            xi_bar_proposal[t] += np.random.normal(0, step_size)
            
            self.xi_bar.assign(xi_bar_proposal.astype(np.float32))
            log_post_proposal = self.log_posterior(X, quantities, market_indices,
                                                   product_indices, market_sizes).numpy()
            
            self.xi_bar.assign(xi_bar_current.astype(np.float32))
            log_post_current = self.log_posterior(X, quantities, market_indices,
                                                  product_indices, market_sizes).numpy()
            
            log_accept = log_post_proposal - log_post_current
            if np.log(np.random.rand()) < log_accept:
                xi_bar_current[t] = xi_bar_proposal[t]
                n_accepts += 1
        
        self.xi_bar.assign(xi_bar_current.astype(np.float32))
        return n_accepts / len(xi_bar_current)
    
    def _mcmc_update_eta(self, X, quantities, market_indices, product_indices, market_sizes):
        """Update η_t using Tailored MH for each market independently.
        
        Paper: π(η_t|•) ∝ L_t(η_t|•) · π(η_t|γ_t)
        Each market t updated independently using TMH.
        """
        eta_current = self.eta.numpy()
        n_accepts = 0
        n_markets = eta_current.shape[0]
        
        for t in range(n_markets):
            eta_t_current = eta_current[t, :]
            dim_eta_t = len(eta_t_current)
            
            # For efficiency, use simplified TMH for η_t
            # Full version would find mode per market, but this is expensive
            # Use adaptive RWMH with market-specific tuning
            
            # Proposal variance adapts to market size
            kappa_eta = 2.38 / np.sqrt(dim_eta_t)
            
            # Compute proposal based on current posterior uncertainty
            # Use spike-slab prior variance as guidance
            gamma_t = self.gamma.numpy()[t, :]
            prior_var_t = gamma_t * self.tau_1_sq + (1 - gamma_t) * self.tau_0_sq
            
            # Proposal: η̃_t ~ N(η_t^{(g)}, κ²_η diag(prior_var))
            eta_t_proposal = eta_t_current + kappa_eta * np.random.randn(dim_eta_t) * np.sqrt(prior_var_t)
            
            # Update eta for this market
            eta_proposal = eta_current.copy()
            eta_proposal[t, :] = eta_t_proposal
            
            # Compute log posterior
            self.eta.assign(eta_proposal.astype(np.float32))
            log_post_proposal = self.log_posterior(X, quantities, market_indices,
                                                   product_indices, market_sizes).numpy()
            
            self.eta.assign(eta_current.astype(np.float32))
            log_post_current = self.log_posterior(X, quantities, market_indices,
                                                  product_indices, market_sizes).numpy()
            
            # MH acceptance (proposal is symmetric given γ_t)
            log_accept = log_post_proposal - log_post_current
            if np.log(np.random.rand()) < log_accept:
                eta_current[t, :] = eta_t_proposal
                n_accepts += 1
        
        self.eta.assign(eta_current.astype(np.float32))
        return n_accepts / n_markets
    
    def _mcmc_update_gamma(self):
        """Update γ_jt using Gibbs sampling."""
        gamma_current = self.gamma.numpy()
        eta_current = self.eta.numpy()
        phi_current = self.phi.numpy()
        
        for t in range(gamma_current.shape[0]):
            for j in range(gamma_current.shape[1]):
                eta_jt = eta_current[t, j]
                
                # Log likelihood under spike and slab
                log_lik_spike = -0.5 * np.log(2 * np.pi * self.tau_0_sq) - 0.5 * eta_jt**2 / self.tau_0_sq
                log_lik_slab = -0.5 * np.log(2 * np.pi * self.tau_1_sq) - 0.5 * eta_jt**2 / self.tau_1_sq
                
                # Prior
                log_prior_0 = np.log(1 - phi_current[t] + 1e-10)
                log_prior_1 = np.log(phi_current[t] + 1e-10)
                
                # Posterior probability
                log_post_0 = log_lik_spike + log_prior_0
                log_post_1 = log_lik_slab + log_prior_1
                
                log_norm = np.logaddexp(log_post_0, log_post_1)
                prob_1 = np.exp(log_post_1 - log_norm)
                
                gamma_current[t, j] = float(np.random.rand() < prob_1)
        
        self.gamma.assign(gamma_current.astype(np.float32))
    
    def _mcmc_update_phi(self):
        """Update φ_t using conjugate Beta posterior."""
        from scipy.stats import beta as beta_dist
        
        gamma_current = self.gamma.numpy()
        phi_current = self.phi.numpy()
        
        for t in range(len(phi_current)):
            n_inclusions = np.sum(gamma_current[t, :])
            n_products = gamma_current.shape[1]
            
            alpha_post = 1.0 + n_inclusions
            beta_post = 1.0 + n_products - n_inclusions
            
            phi_current[t] = beta_dist.rvs(alpha_post, beta_post)
        
        self.phi.assign(phi_current.astype(np.float32))
    
    def _compute_mcmc_summaries(self, samples, accept_counts):
        """Compute posterior summaries from MCMC samples."""
        summaries = {}
        
        for param_name in ['beta_mean', 'log_sigma', 'xi_bar', 'eta', 'gamma', 'phi']:
            samples_array = np.array(samples[param_name])
            
            summaries[param_name] = {
                'mean': np.mean(samples_array, axis=0),
                'std': np.std(samples_array, axis=0),
                'median': np.median(samples_array, axis=0),
                'q025': np.percentile(samples_array, 2.5, axis=0),
                'q975': np.percentile(samples_array, 97.5, axis=0)
            }
        
        # Compute σ = exp(log_sigma)
        sigma_samples = np.exp(np.array(samples['log_sigma']))
        summaries['sigma'] = {
            'mean': np.mean(sigma_samples, axis=0),
            'std': np.std(sigma_samples, axis=0),
            'median': np.median(sigma_samples, axis=0),
            'q025': np.percentile(sigma_samples, 2.5, axis=0),
            'q975': np.percentile(sigma_samples, 97.5, axis=0)
        }
        
        # Acceptance rates
        n_post_burnin = len(samples['beta_mean'])
        summaries['acceptance_rates'] = {
            key: val / self.n_iterations for key, val in accept_counts.items()
        }
        
        return summaries
    
    def _print_mcmc_results(self, summaries):
        """Print MCMC estimation results."""
        print(f"\n{'='*70}")
        print("MCMC ESTIMATION RESULTS")
        print(f"{'='*70}\n")
        
        print("β̄ (mean coefficients):")
        for i, (mean, std) in enumerate(zip(summaries['beta_mean']['mean'],
                                            summaries['beta_mean']['std'])):
            print(f"  β̄[{i}]: {mean:8.4f} (±{std:.4f})")
        
        print("\nσ (random coefficient std):")
        for i, (mean, std) in enumerate(zip(summaries['sigma']['mean'],
                                            summaries['sigma']['std'])):
            print(f"  σ[{i}]: {mean:8.4f} (±{std:.4f})")
        
        print(f"\nSparsity: {1 - np.mean(summaries['gamma']['mean']):.2%}")
        print(f"φ (inclusion prob): {np.mean(summaries['phi']['mean']):.4f}")
        
        print("\nAcceptance Rates:")
        for key, val in summaries['acceptance_rates'].items():
            print(f"  {key}: {val:.3f}")
        
        print(f"\n{'='*70}\n")


def create_sparse_demand_model(
    n_features=2,
    n_markets=None,
    n_products_per_market=None,
    n_simulations=200,
    tau_0_sq=0.001,
    tau_1_sq=1.0
):
    """Factory function to create Sparse Demand Model.
    
    Parameters
    ----------
    n_features : int
        Number of product features (including price)
    n_markets : int, optional
        Number of markets
    n_products_per_market : int, optional
        Number of products per market
    n_simulations : int
        Number of Monte Carlo draws
    tau_0_sq : float
        Spike variance
    tau_1_sq : float
        Slab variance
        
    Returns
    -------
    SparseDiscreteChoiceModel
        Initialized model
    """
    return SparseDiscreteChoiceModel(
        n_features=n_features,
        n_markets=n_markets,
        n_products_per_market=n_products_per_market,
        n_simulations=n_simulations,
        tau_0_sq=tau_0_sq,
        tau_1_sq=tau_1_sq
    )
