# AGENTS.md - TechChallenge1ChurnTelco

ML pipeline for telecom churn prediction. Python 3.12+, PyTorch MLP, scikit-learn baselines, MLflow tracking, FastAPI inference.

## Philosophy: Simplicity First

- **Prefer simple solutions** - avoid over-engineering
- **Question every addition** - does this really need to be added?
- **Function over form** - working code beats perfect architecture
- **Less is more** - fewer lines, fewer files, fewer dependencies

## Commands

Uses `uv` as package manager. Always prefix Python commands with `uv run`.

```bash
# Setup
make setup                    # uv sync + pre-commit install + DVC setup
uv sync                       # Install deps (add --no-dev for CI/Docker)

# Testing (80% coverage minimum enforced)
make test                     # Full test suite with coverage
uv run pytest tests/ -v       # Run all tests
uv run pytest -m fast -v      # Quick tests only
uv run pytest -m slow -v      # Integration tests only

# Quality
make lint                     # Check with ruff
make format                   # Format with ruff
uv run mypy src/              # Type check (strict mode)

# MLflow (requires .env)
make docker-up                # Start MLflow + PostgreSQL + MinIO
make docker-down              # Stop containers
```

## Code Style (Strict)

- **Line length:** 79 characters (not 88)
- **Python:** 3.12+ with `from __future__ import annotations` always
- **Type hints:** Required on all functions (`disallow_untyped_defs = true`)
- **Import order:** stdlib → third-party → local (enforced by ruff I rule)
- **Use `TYPE_CHECKING`** for imports only needed for type hints

### Patterns

```python
# Dataclasses for config
@dataclass(frozen=True)
class Config:
    value: str

# Protocols for interfaces
class Trainer(Protocol):
    def train(self, data: Data) -> Model: ...

# Error handling
except SpecificException as e:  # Be specific
except Exception as e:  # noqa: BLE001  # Rare, mark with noqa
```

## Project Structure

```
src/
├── api/           # FastAPI (empty - .gitkeep)
├── data/          # Data loading (empty - .gitkeep)
├── eda/           # EDA helpers (empty - .gitkeep)
├── features/      # Feature pipelines (empty - .gitkeep)
├── inference/     # Prediction service (empty - .gitkeep)
├── pipelines/     # End-to-end scripts (empty - .gitkeep)
├── schemas/       # Pydantic models (empty - .gitkeep)
└── training/      # Model training (empty - .gitkeep)

data/              # DVC-tracked datasets
├── raw/           # Original data
└── processed/     # Transformed data

models/            # Saved model artifacts
tests/             # pytest tests (test_*.py)
notebooks/         # Exploration only - NO artifacts
docs/              # Documentation
```

## Critical Rules

1. **NO artifacts from notebooks** - notebooks are exploration only; final artifacts must come from `src/pipelines/` scripts
2. **All code needs type hints** - mypy runs in strict mode
3. **Coverage minimum 80%** - enforced in CI via pyproject.toml
4. **Test markers:** `@pytest.mark.fast` for quick tests, `@pytest.mark.slow` for integration
5. **NO emojis anywhere** - use ASCII text equivalents (see Text Style section below)

## Text and Documentation Style

**NO emojis in any file** - this includes source code, documentation, comments, commit messages, and shell scripts. Use ASCII text equivalents:

| Instead of | Use |
|------------|-----|
| Checkmark | `[OK]`, `Success:`, `Done:` |
| Cross | `[ERROR]`, `Error:` |
| Warning | `[WARN]`, `Warning:` |
| Whale | `Docker:` |
| Rocket | `Starting...`, `Launching...` |
| Lightbulb | `Tip:`, `Note:` |
| Chart | `Results:`, `Metrics:` |
| Target | `Goal:`, `Next:` |
| Wrench | `Config:`, `Setup:` |
| Folder | `Directory:`, `Folder:` |
| Clipboard | `List:`, `Summary:` |

**NO emoji in:**
- Source code (Python files)
- Documentation (README, AGENTS.md, etc)
- Comments (inline or block)
- Commit messages
- Shell scripts (Makefile, .sh files)
- Configuration files

## Environment Setup

```bash
# Required before running MLflow or pipeline scripts
cp .env.example .env          # Then edit with your values
```

Key env vars:
- `MLFLOW_TRACKING_URI=http://localhost:5000`
- `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`
- `MLFLOW_EXPERIMENT_NAME=tech-challenge-default`
- `DVC_ONEDRIVE_REMOTE_URL=` (set during `make setup`)

## MLflow Integration

```python
from dotenv import load_dotenv
import mlflow

load_dotenv()

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

## Dependencies

- Add prod: `uv add <package>`
- Add dev: `uv add --dev <package>`
- Lock file: `uv.lock` (must be committed)

Key deps: fastapi, pandas, scikit-learn, torch, mlflow, pytest, ruff, mypy
