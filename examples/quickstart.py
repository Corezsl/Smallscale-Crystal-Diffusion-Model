"""Example entry script for a minimal crystal diffusion workflow."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from crystal_diffusion.config import load_config
from crystal_diffusion.data import make_synthetic_crystal_dataset
from crystal_diffusion.diffusion import CrystalDiffusion


def main() -> None:
    """Run a tiny demonstration of the diffusion primitives."""
    config = load_config()
    dataset = make_synthetic_crystal_dataset(
        num_structures=2,
        num_atoms=4,
        cutoff=config["data"]["cutoff"],
    )
    sample = dataset[0]
    diffusion = CrystalDiffusion(
        timesteps=config["model"]["timesteps"],
        beta_start=config["model"]["beta_start"],
        beta_end=config["model"]["beta_end"],
        atom_latent_dim=config["model"]["atom_latent_dim"],
        num_elements=config["model"]["num_elements"],
    )
    x_0 = Data(
        lattice=sample.lattice,
        frac_coords=sample.frac_coords,
        atom_types=sample.atom_types,
    )
    noisy = diffusion.q_sample(x_0, t=10)
    print(f"Sample lattice shape: {tuple(noisy.lattice.shape)}")
    print(f"Sample atomic latent shape: {tuple(noisy.atom_latent.shape)}")
    print(f"Fractional coordinates in range: {torch.min(noisy.frac_coords).item():.3f} to {torch.max(noisy.frac_coords).item():.3f}")
    print("Synthetic crystal diffusion example completed successfully.")


if __name__ == "__main__":
    main()
