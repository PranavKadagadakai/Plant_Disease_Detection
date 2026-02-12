# AGENTS.md - Coding Guidelines for Plant Disease Detection

This document provides guidelines for AI agents working in this PyTorch-based plant disease classification repository.

## Build and Package Commands

This project uses `uv` (Python package manager) with `pyproject.toml`:

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Run a script with dependencies
uv run python train.py

# Activate virtual environment
source .venv/bin/activate
```

## Running Tests and Scripts

**Note:** No formal test framework (pytest) is configured. Use direct script execution:

```bash
# Train the model
python train.py

# Evaluate on test set
python evaluate.py

# Make a single prediction
python predict.py

# Run any script individually
python <script_name>.py
```

### Recommended: Add pytest for testing

```bash
uv add --dev pytest
pytest tests/                    # Run all tests
pytest tests/test_model.py       # Run single test file
pytest tests/test_model.py::test_build_model  # Run single test
```

## Code Style Guidelines

### Import Conventions

- Use direct imports over module imports:
  ```python
  # Good
  from torch.utils.data import DataLoader
  from torchvision import models
  
  # Avoid
  import torch.utils.data
  ```

- Group imports in order:
  1. Standard library
  2. Third-party packages
  3. Local modules

- Separate groups with blank lines

### Naming Conventions

- **Constants:** UPPER_CASE at module level
  ```python
  BATCH_SIZE = 32
  EPOCHS = 10
  MODEL_PATH = "plant_disease_model.pth"
  ```

- **Functions/Variables:** snake_case
  ```python
  def build_model(num_classes):
      train_loader = DataLoader(...)
  ```

- **Model Files:** Descriptive names (`model.py`, `train.py`, `evaluate.py`)

### Code Structure

- **Separate configuration from logic:** Keep CONFIG section at top of scripts
- **Device handling:** Always check for CUDA availability
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- **Model definition:** Keep in separate module (`model.py`)
- **Avoid execution code in imported modules:** Use `if __name__ == "__main__":`

### Error Handling

- Check for file existence before loading:
  ```python
  import os
  if not os.path.exists(MODEL_PATH):
      raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
  ```

- Use context managers for resource handling:
  ```python
  with torch.no_grad():
      outputs = model(images)
  ```

### Type Hints (Recommended)

Add type hints for function signatures:
```python
def build_model(num_classes: int) -> nn.Module:
    ...
```

## Recommended Tools to Add

Install these for better code quality:

```bash
# Linting and formatting
uv add --dev ruff
uv run ruff check .           # Lint all files
uv run ruff check --fix .     # Auto-fix issues
uv run ruff format .          # Format all files

# Type checking
uv add --dev mypy
uv run mypy .                 # Type check

# Testing
uv add --dev pytest pytest-cov
uv run pytest --cov=. tests/  # Run tests with coverage
```

## Project-Specific Patterns

### Data Loading

Always use the same transforms as training:
```python
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Model Loading

Always specify `map_location` for device compatibility:
```python
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
```

### Split Indices

Training saves indices to `splits/` directory. Always load these for consistent evaluation:
```python
test_idx = torch.load("splits/test_idx.pt")
test_ds = Subset(dataset, test_idx)
```

## Common Tasks

### Add a new evaluation metric

Edit `evaluate.py` and add metrics after line 64:
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")
```

### Change model architecture

Edit `model.py` and update the `build_model` function. Keep the same interface.

### Add data augmentation

Edit `train.py` transform pipeline:
```python
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
