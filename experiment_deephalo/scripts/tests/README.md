## Test Suite for DeepHalo Experiments Scripts

This test suite includes unit tests and integration tests for the training scripts.

## Running Tests

### Run all tests:
```bash
cd scripts
pytest tests/ -v
```

### Run only unit tests:
```bash
pytest tests/unit/ -v
```

### Run only integration tests:
```bash
pytest tests/integration/ -v
```

### Run specific test file:
```bash
pytest tests/unit/test_data_utils.py -v
```

## Test Structure

- `tests/unit/` - Unit tests for individual functions and classes
  - `test_data_utils.py` - Tests for data loading and processing
  - `test_metrics.py` - Tests for RMSE computation and callbacks
  - `test_plotting.py` - Tests for visualization functions

- `tests/integration/` - Integration tests for complete workflows
  - `test_training_pipeline.py` - End-to-end training pipeline tests

## Requirements

Install test dependencies:
```bash
pip install pytest
```
