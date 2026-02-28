# Discrete Choice Model

This project extends the [Choice-Learn](https://github.com/artefactory/choice-learn) library with new models for discrete choice/demand prediction.

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

## DeepHalo Model (Featureless and Feature-based)

### New Files in `choice-learn`

- Model implementation: `choice-learn/choice_learn/models/deephalo_{featureless,feature}.py`
- Unit tests: `choice-learn/tests/unit_tests/models/test_deephalo_{featureless,feature}.py`
- Integration tests: `choice-learn/tests/integration_tests/models/test_deephalo_{featureless,feature}.py`
- Example notebook: `choice-learn/notebooks/models/deephalo_{featureless,feature}.ipynb`


### Reproduce DeepHalo Featureless Synthetic Experiments

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