# Discrete Choice Model

This project extends the [Choice-Learn](https://github.com/artefactory/choice-learn) library with new deep learning models for discrete choice prediction.

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

## DeepHalo Featureless Model

### New Files Added

1. Model implementation: `choice-learn/choice_learn/models/deephalo_featureless.py`
2. Tests (run with `pytest /path/to/test_script.py`):
    - Unit tests: `choice-learn/tests/unit_tests/models/test_deephalo_featureless.py`
    - Integration tests: `choice-learn/tests/integration_tests/models/test_deephalo_featureless.py`
3. Example usage notebook: `choice-learn/notebooks/models/deephalo_featureless.ipynb`
4. Synthetic data generation: `data/synthetic_data_generation.py`