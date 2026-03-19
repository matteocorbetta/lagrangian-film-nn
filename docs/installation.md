# Installation And Docker

This repository uses `uv` as the primary environment manager, with reproducibility anchored by `.python-version`, `pyproject.toml`, and `uv.lock`.

## Local Setup With `uv`

Install the locked environment from the repository root:

```bash
uv sync
```

Run commands inside the managed environment:

```bash
uv run python -c "import jax, equinox, optax"
```

## Docker Smoke Test

The Docker setup is intentionally minimal. It is a Linux CPU-only smoke test meant to verify that the repository builds and the core Python stack imports correctly inside a clean container.

Build the image from the repository root:

```bash
docker build -t lagrangiannn .
```

Run the default smoke test:

```bash
docker run --rm lagrangiannn
```

The default container command runs:

```bash
uv run python -c "import jax, equinox, optax; print('smoke test ok')"
```

On macOS, Docker Desktop runs Linux containers inside a lightweight VM. That means this verifies Linux-in-Docker behavior, not native macOS execution. A local macOS `Python 3.14.x` virtual environment is a useful signal, but it does not replace validating the Linux container environment.

If you want to manually test the first script-oriented workflow step after the smoke test, you can run:

```bash
docker run --rm -it lagrangiannn uv run python src/data/generate_dataset.py
```
