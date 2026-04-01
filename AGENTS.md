# AGENTS.md - Coding Guidelines for TechChallenge1ChurnTelco

This file provides essential information for AI coding agents working on this ML project.

## Project Overview

This is a Python 3.12+ machine learning project for FIAP's MLE post-graduation Tech Challenge. It uses MLflow for experiment tracking, FastAPI for serving, and scikit-learn for modeling.

## Core Philosophy: Simplicity First

**Code must always be simple, concise, and functional.**

When adding or changing anything:
- **Prefer simplicity** - Avoid over-engineering and unnecessary complexity
- **Do the minimum** - Plan to write only what's needed to solve the problem
- **Be direct** - No fluff, no unnecessary abstractions, no "just in case" features
- **Function over form** - Working code beats perfect architecture
- **Less is more** - Fewer lines, fewer files, fewer dependencies = better
- **Question every addition** - Does this really need to be added? Can it be simpler?

**Examples of what to avoid:**
- Complex class hierarchies when functions suffice
- Abstraction layers that don't provide immediate value
- Generic "frameworks" for simple tasks
- Future-proofing for hypothetical scenarios
- Boilerplate that doesn't add functionality

**When in doubt, choose the simpler option.**

## Build/Lint/Test Commands

All commands use `uv` as the package manager:

```bash
# Setup environment
make setup                    # Install deps and pre-commit hooks
uv sync                       # Install all dependencies
uv sync --no-dev             # Production only (CI/Docker)

# Testing
make test                     # Run all tests with coverage
uv run pytest tests/ -v      # Run all tests (verbose)
uv run pytest tests/test_file.py -v              # Single test file
uv run pytest tests/test_file.py::test_func -v   # Single test function
uv run pytest -m fast -v     # Run only fast tests
uv run pytest -m slow -v     # Run only slow tests

# Coverage requirements: 80% minimum on src/
# Coverage config is in pyproject.toml

# Linting and Formatting
make lint                    # Check code with ruff
make format                  # Format code with ruff
uv run ruff check .          # Manual lint check
uv run ruff check . --fix    # Auto-fix issues
uv run ruff format .         # Manual format

# Type Checking
uv run mypy src/             # Check types (strict mode enabled)

# Pre-commit
uv run pre-commit run --all-files     # Run all hooks
uv run pre-commit run ruff --all-files # Run specific hook

# MLflow (Docker)
make docker-up               # Start MLflow + PostgreSQL + MinIO
docker-compose -f docker/docker-compose.yml up -d
make docker-down             # Stop all containers
```

## Code Style Guidelines

### Python Version and Imports
- Use Python 3.12+ features (requires-python = ">=3.12,<3.14")
- Always use `from __future__ import annotations` at the top
- Import order: stdlib → third-party → local (enforced by ruff I rule)
- Use `TYPE_CHECKING` for imports only needed for type hints

### Type Hints (Strict)
- **All functions must have type hints** (`disallow_untyped_defs = true`)
- Use modern syntax: `list[str]`, `dict[str, Any]`, `str | None`
- Use `NDArray[Any]` from `numpy.typing` for array types
- Use Protocols for interface definitions

### Formatting
- Line length: **79 characters** (PEP 8 standard)
- Use ruff for both linting and formatting
- Ruff rules enabled: I (imports), F (Pyflakes), E/W (pycodestyle), PL (pylint), PT (pytest)

### Naming Conventions
- Classes: `PascalCase` (e.g., `ExperimentRunner`, `ModelConfig`)
- Functions/variables: `snake_case` (e.g., `run_experiment`, `model_name`)
- Constants: `UPPER_CASE` (e.g., `HTTP_OK`, `DEFAULT_PORT`)
- Private: `_leading_underscore` for internal use
- Enums: Use `auto()` for values when appropriate

### Code Structure
- Use `@dataclass` for configuration objects (prefer `frozen=True`)
- Use `@dataclass` for entities with `field(default_factory=list)` for mutable defaults
- Use Protocols for dependency injection and interfaces
- Prefer static methods in trainer/utility classes
- Organize imports: stdlib, third-party, local with blank lines between

### Error Handling
- Use specific exceptions when possible
- Use `try/except` with context managers
- Log errors appropriately using logging module
- For expected errors: `except SpecificException as e:`
- For general catching (use sparingly): `except Exception as e:  # noqa: BLE001`
- Return boolean status for validation functions

### Documentation
- All modules need docstrings with triple quotes
- All public functions need docstrings
- Use Google-style or standard docstrings
- Add type hints instead of documenting types in docstrings

### Testing
- Test files: `tests/test_*.py`
- Test markers: `@pytest.mark.fast` for quick tests, `@pytest.mark.slow` for integration
- Minimum coverage: 80% on `src/`
- Use pytest fixtures for shared setup
- Mock external services (MLflow, databases) in unit tests

### MLflow Integration
- Load config from environment using `MLflowConfig.from_env()`
- Use `python-dotenv` for `.env` file support
- Always use context managers: `with mlflow.start_run():`
- Log params with `mlflow.log_param()`, metrics with `mlflow.log_metric()`
- Log models with appropriate flavor: `mlflow.sklearn.log_model()`

### Project Structure Rules
- **NO artifacts in notebooks** - notebooks are for exploration only
- Final artifacts (models, datasets) go in `src/pipelines/` as parameterized scripts
- Data goes in `data/`, models in `models/`, docs in `docs/`
- Source code organized: `api/`, `data/`, `features/`, `training/`, `inference/`, `pipelines/`, `schemas/`

### Environment Management
- Use `.env` file for local configuration (copy from `.env.example`)
- Never commit `.env` files
- Use `python-dotenv` for loading environment variables
- Docker Compose uses `.env` file automatically

### Dependencies
- Production deps: listed in `[project] dependencies`
- Dev deps: listed in `[dependency-groups] dev`
- Use `uv add <package>` to add production dependencies
- Use `uv add --dev <package>` to add dev dependencies
- Lock file `uv.lock` must be committed

### Git Workflow
- Pre-commit hooks run automatically on commit
- CI autofixes PRs with pre-commit
- Use conventional commit messages

## Project Architecture

### Directory Structure
```
src/
├── api/           # FastAPI application and endpoints
├── data/          # Data loading and validation
├── eda/           # Exploratory data analysis tools
├── features/      # Feature engineering pipelines
├── inference/     # Model serving and prediction
├── pipelines/     # End-to-end training pipelines
├── schemas/       # Pydantic models and data contracts
└── training/      # Model training and evaluation
data/              # Raw and processed datasets
models/            # Saved model artifacts
tests/             # Unit and integration tests
notebooks/         # Jupyter notebooks for exploration
docs/              # Documentation and reports
```

### Key Design Patterns
- **Configuration as Code**: Use dataclasses for all config objects
- **Protocol-based Interfaces**: Define contracts with typing.Protocol
- **Static Methods**: Prefer `@staticmethod` for utility functions
- **Context Managers**: Use `with` statements for resource management
- **Immutable Data**: Use `frozen=True` in dataclasses where possible
