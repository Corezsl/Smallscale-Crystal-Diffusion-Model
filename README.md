# Smallscale Crystal Diffusion Model

[![CI](https://github.com/Corezsl/Smallscale-Crystal-Diffusion-Model/actions/workflows/test.yml/badge.svg)](https://github.com/Corezsl/Smallscale-Crystal-Diffusion-Model/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Smallscale Crystal Diffusion Model is an early-stage Python project for training and evaluating diffusion models over periodic crystal structures. The codebase models a crystal as a joint state made of:

- a 3x3 lattice matrix,
- fractional atomic coordinates on a unit torus,
- learned atom-latent representations for atomic types.

This repository is intended for research and experimentation in materials generation and structure modeling. It includes dataset utilities, a forward diffusion implementation, training scaffolding, generation and evaluation helpers, and a lightweight Streamlit interface.

## Status

This repository is a research prototype rather than a production-ready library. It provides the core building blocks for a crystal diffusion pipeline, but you will still need to provide a model architecture, dataset collection, and training workflow matched to your application.

## Why this project exists

The goal is to model periodic structures using a diffusion process over:

1. the lattice matrix,
2. fractional coordinates,
3. atom-type latent representations.

This is useful for learning generative distributions over crystal structures and evaluating the physical plausibility of generated examples.

## Features

- Crystal graph representation with Pymatgen and PyTorch Geometric
- Forward diffusion over lattice, fractional coordinates, and atom latent variables
- Training loop scaffold using AdamW and MSE objectives
- Reverse diffusion sampling utilities for crystal generation
- Structure evaluation and geometry validation helpers
- Materials Project download helper
- Optional Streamlit GUI for prototyping and inspection
- Simple configuration loader for project defaults
- Synthetic crystal dataset utilities for demos and tests

## Installation

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional development tools:

```bash
python -m pip install -e ".[dev]"
```

Optional app dependencies:

```bash
python -m pip install -e ".[app]"
```

## Quick start

### Basic dataset and diffusion usage

```python
import torch
from torch_geometric.data import Data
from crystal_diffusion.diffusion import CrystalDiffusion

x_0 = Data(
    lattice=torch.eye(3),
    frac_coords=torch.tensor([[0.2, 0.5, 0.9]], dtype=torch.float32),
    atom_types=torch.tensor([14], dtype=torch.long),
)

diffusion = CrystalDiffusion(timesteps=1000, atom_latent_dim=64)
noisy = diffusion.q_sample(x_0, t=500)
print(noisy.lattice.shape)
print(noisy.frac_coords.shape)
print(noisy.atom_latent.shape)
```

### Run the test suite

```bash
python -m pytest
```

### Download structures from Materials Project

Set your API key:

```bash
export MP_API_KEY="your-api-key"
```

Then run:

```bash
python scripts/download_mp.py
```

### Run the quick example script

```bash
python examples/quickstart.py
```

## Command-line interface

The package now includes a light CLI for common actions:

```bash
smallscale-crystal-download --help
smallscale-crystal-train --help
smallscale-crystal-generate --help
smallscale-crystal-evaluate --help
```

These commands provide usage guidance and are intended to support a research workflow without requiring a fully packaged training pipeline.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements.txt
├── config.yaml
├── docs/
│   └── overview.md
├── examples/
│   ├── README.md
│   └── quickstart.py
├── scripts/
│   ├── app.py
│   ├── download_mp.py
│   ├── evaluate.py
│   ├── generate.py
│   └── train.py
├── src/
│   └── crystal_diffusion/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── synthetic.py
│       ├── diffusion/
│       │   ├── __init__.py
│       │   └── diffusion.py
│       └── geometry/
│           └── __init__.py
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_diffusion.py
│   ├── test_package.py
│   └── test_synthetic_dataset.py
└── .github/
    └── workflows/
        └── test.yml
```

## Training workflow

The repository includes a training helper in `scripts/train.py`. The intended flow is to:

1. build a `CrystalDataset` from Pymatgen structures,
2. create a `CrystalGNN` model that accepts `(noisy_batch, timesteps)`,
3. pass the model to the training entry point,
4. optimize against the forward-diffusion noise targets.

The project expects model outputs to provide the three diffusion target channels:

- `lattice`
- `frac_coords`
- `atom_latent`

## Inference and evaluation

- `scripts/generate.py` contains reverse diffusion utilities.
- `scripts/evaluate.py` can decode atom latents, reconstruct structures, and run geometry/energy checks.
- `scripts/app.py` provides an optional Streamlit GUI for interactive inspection.
- `examples/quickstart.py` demonstrates the simplest end-to-end usage pattern.

## Configuration

A sample configuration file is included at `config.yaml`. The project also exposes a configuration loader in `crystal_diffusion.config.load_config()` so that external scripts can read a consistent set of defaults.

Example:

```python
from crystal_diffusion.config import load_config

config = load_config()
print(config["training"]["epochs"])
```

## Synthetic datasets

The project includes a lightweight synthetic crystal-data utility for quick experimentation and tests. This is useful when you want to validate code paths without accessing external Materials Project data.

```python
from crystal_diffusion.data import make_synthetic_crystal_dataset

dataset = make_synthetic_crystal_dataset(num_structures=4, cutoff=5.0)
print(len(dataset))
```

## Known limitations

- This project is not a complete end-to-end materials generation toolkit yet.
- Model architecture and training procedures still need to be supplied by the user for a full application.
- Some evaluation checks may fail for unsupported chemical systems or unrealistic generated compositions.
- The repository does not yet provide a large benchmark dataset or a fully packaged training pipeline.
- The example CLI commands are guidance-oriented and intentionally lightweight.

## Contributing

Contributions are welcome. Please see `CONTRIBUTING.md` for the development flow, code standards, and pull-request expectations.

## Changelog

See `CHANGELOG.md` for a record of project updates.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## References

This project is built around ideas from diffusion models for continuous representations of crystal structures and is designed to work naturally with Pymatgen and PyTorch Geometric.
