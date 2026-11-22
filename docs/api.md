# API

This page documents the reusable modules in the repository. The project is primarily script-driven, but several modules define interfaces that are useful as importable components.

## API Overview

The most reusable modules are:

- `lagrangian_film_nn.lnn`: model definition and the `LagrangianNN` entrypoint
- `lagrangian_film_nn.data`: analytical system helpers and HDF5 dataset utilities
- training and simulation helpers in `lagrangian_film_nn.train_utils`, `lagrangian_film_nn.simulate`, and `lagrangian_film_nn.losses`

The API documentation below focuses on importable functions and classes rather than `__main__` script blocks.

## Core Model

::: lagrangian_film_nn.lnn

::: lagrangian_film_nn.lnn.model

## Analytical System And Dataset Utilities

::: lagrangian_film_nn.data

::: lagrangian_film_nn.data.doublependulum

::: lagrangian_film_nn.data.utils

## Training And Simulation Helpers

::: lagrangian_film_nn.train_utils

::: lagrangian_film_nn.simulate

::: lagrangian_film_nn.losses
