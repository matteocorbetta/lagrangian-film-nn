"""Project-wide constants shared across the lagrangian_film_nn package."""

from pathlib import Path


# CONSTANTS
# =========
GRAVITY = 9.806

# Repository root: src/lagrangian_film_nn/utils.py -> parents[2].
# Valid for editable installs (`pip install -e .` / `uv sync`), where __file__ stays in the source tree.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
