# Installation

This page covers environment setup, dependency assumptions, and basic verification steps for building and using the documentation site.

## Environment Assumptions

The repository currently declares `requires-python = ">=3.14"` in `pyproject.toml`. That is an unusually new version requirement and should be treated as a project constraint to verify against your local JAX stack before relying on it for long-term reproducibility.

The lockfile `uv.lock` suggests that the preferred environment manager is [`uv`](https://github.com/astral-sh/uv). The commands below assume you are running from the repository root.

## Preferred Setup With `uv`

Install dependencies from the project metadata and lockfile:

```bash
uv sync
```

Run Python commands inside the managed environment:

```bash
uv run python -c "import jax, equinox, optax"
```

Serve the documentation locally:

```bash
uv run mkdocs serve
```

Build the static documentation site:

```bash
uv run mkdocs build
```

## Alternative Setup With `pip`

If you are not using `uv`, an editable install is the closest equivalent:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If the editable install does not include documentation extras in your workflow, install the docs stack explicitly:

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
```

## Dependency Overview

Core ML and numerics:

- `jax`
- `equinox`
- `optax`
- `numpy`

Data handling and plotting:

- `h5py`
- `matplotlib`

Documentation:

- `mkdocs`
- `mkdocs-material`
- `mkdocstrings[python]`

## JAX Installation Note

`jax` installation can vary by platform and accelerator support. If you are targeting GPU or TPU execution, verify that your environment installs the correct JAX and `jaxlib` combination for your platform. The repository itself does not currently document accelerator-specific setup beyond listing `jax` as a dependency.

## Project Layout Assumptions

Most scripts are written as repo-root entrypoints and import modules from `src/`. In practice this means:

- run commands from the repository root,
- prefer `uv run python ...` or an activated environment from the root directory,
- expect some scripts to rely on direct imports such as `from train_utils import ...` or `from data import ...`.

The API documentation also relies on MkDocs being able to import modules from `src/`.

## Verify The Environment

Minimal import smoke test:

```bash
uv run python -c "import jax, equinox, optax"
```

Check that MkDocs is installed in the active environment:

```bash
uv run mkdocs --version
```

Check that the documentation builds:

```bash
uv run mkdocs build
```

## Current Repository Inconsistency

The package metadata currently uses the placeholder project name `jax-practice`, while the docs and site branding use `Lagrangian-FiLM NN`. This documentation follows the repository branding visible in `mkdocs.yml`, but the mismatch in `pyproject.toml` remains a project-level cleanup item rather than a docs-only issue.
