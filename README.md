# Smallscale Crystal Diffusion Model

Smallscale Crystal Diffusion Model is an early-stage Python project for training and evaluating diffusion models over periodic crystal structures. The codebase models a crystal as a joint state made of:

- a 3x3 lattice matrix,
- fractional atomic coordinates on a unit torus,
- learned atom-latent representations for atomic types.

The project is intended for research and experimentation in materials generation and structure modeling. It includes dataset utilities, a forward diffusion implementation, training scaffolding, generation/evaluation helpers, and a lightweight Streamlit interface.

## Status

This repository is a research prototype rather than a production-ready library. It provides the core building blocks for a crystal diffusion pipeline, but you will still need to provide a model architecture, dataset collection, and training workflow appropriate to your application.

## Features

- Crystal graph representation with Pymatgen and PyTorch Geometric
- Forward diffusion over lattice, fractional coordinates, and atom latent variables
- Training loop scaffold using AdamW and MSE objectives
- Reverse diffusion sampling utilities for crystal generation
- Structure evaluation and geometry validation helpers
- Materials Project download helper
- Optional Streamlit GUI for prototyping and inspection

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

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── app.py
│   ├── download_mp.py
│   ├── evaluate.py
│   ├── generate.py
│   └── train.py
├── src/
│   └── crystal_diffusion/
│       ├── __init__.py
│       ├── data/
│       ├── diffusion/
│       └── geometry/
├── tests/
│   ├── test_dataset.py
│   ├── test_diffusion.py
│   └── test_package.py
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

## Known limitations

- This project is not a complete end-to-end materials generation toolkit yet.
- Model architecture and training procedures still need to be supplied by the user for a full application.
- Some evaluation checks may fail for unsupported chemical systems or unrealistic generated compositions.
- The repository does not yet provide a large benchmark dataset or a fully packaged training pipeline.

## Configuration

A sample configuration file is included at `config.yaml`. It shows the expected configuration fields for model, training, and Materials Project settings.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contributing

Contributions are welcome. If you plan to extend the project, please keep the code modular, add tests for new functionality, and document any new inputs or outputs clearly.

## References

This project is built around ideas from diffusion models for continuous representations of crystal structures and is designed to work naturally with Pymatgen and PyTorch Geometric.
