# Discrete Choice Model

This repository extends the [Choice-Learn](https://github.com/artefactory/choice-learn) library with state-of-the-art models for discrete choice and demand prediction. It provides TensorFlow implementations, example notebooks, and experiment scripts (including synthetic data generation).

Included models:
- **DeepHalo**: [Deep Context-Dependent Choice Model](https://openreview.net/forum?id=bXTBtUjb0c), supporting both featureless and feature-based variants.
- **Sparse Demand Model**: [Estimating Discrete Choice Demand Models with Sparse Market-Product Shocks](https://arxiv.org/abs/2501.02381), enabling identification without instrumental variables.


## Environment

- **Python**: 3.10.x
- **Required Packages**:
  - `tensorflow` (2.20)
  - `pandas`
  - `matplotlib`
  - `tqdm`
  - `pytest`

Install requirements:
```bash
pip install tensorflow pandas matplotlib tqdm pytest
```

---

## DeepHalo Model

We provide both featureless and feature-based implementations of DeepHalo, adapted from the [official PyTorch repository](https://github.com/Asimov-Chuang/DeepHalo). The featureless version is used for synthetic data experiments.


### New Files in `choice-learn`

- Model implementation: `choice-learn/choice_learn/models/deephalo_{featureless,feature}.py`
- Tests (run with `pytest /path/to/test_script.py`):
  - Unit tests: `choice-learn/tests/unit_tests/models/test_deephalo_{featureless,feature}.py`
  - Integration tests: `choice-learn/tests/integration_tests/models/test_deephalo_{featureless,feature}.py`
- Example notebook: `choice-learn/notebooks/models/deephalo_{featureless,feature}.ipynb`


### Reproduce Synthetic Data Experiments

1. Enter the experiment directory:
    ```bash
    cd experiment_deephalo
    ```

2. Generate synthetic data:
    ```bash
    python data/synthetic_data_generation.py
    ```

3. Run the experiment (featureless model example):
    ```bash
    python scripts/train_deephalo_featureless.py
    ```

4. Review results and analysis:
    - Outputs are saved in the `results/` directory


---

## Sparse Demand Model

We introduce the Sparse Demand Model, which estimates discrete choice demand with sparse market-product shocks. This approach leverages Bayesian inference and shrinkage priors to identify and estimate model parameters without relying on instrumental variables.


### New Files in `choice-learn`

- Model implementation: `choice-learn/choice_learn/models/sparse_demand_model.py`
- Tests (run with `pytest /path/to/test_script.py`):
  - Unit tests: `choice-learn/tests/unit_tests/models/test_sparse_demand_model.py`
  - Integration tests: `choice-learn/tests/integration_tests/models/test_sparse_demand_model.py`
- Example notebook: `choice-learn/notebooks/models/sparse_demand_model.ipynb`


### Reproduce Monte Carlo Simulation Experiments

1. Enter the experiment directory:
    ```bash
    cd experiment_sparse_demand
    ```

2. Generate synthetic data:
    ```bash
    python data/monte_carlo_data_generator.py
    ```