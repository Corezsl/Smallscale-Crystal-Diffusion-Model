"""Generate periodic crystal graphs with a trained noise-prediction GNN."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from crystal_diffusion.diffusion import CrystalDiffusion


def _prediction(output: Mapping[str, Tensor] | Any, name: str) -> Tensor:
    value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
    if value is None:
        raise ValueError(f"model output is missing noise prediction: {name}")
    return value


def _initial_lattice(device: torch.device, dtype: torch.dtype) -> Tensor:
    lengths = torch.empty(3, device=device, dtype=dtype).uniform_(3.0, 8.0)
    return torch.diag(lengths) + 0.05 * torch.randn((3, 3), device=device, dtype=dtype)


def p_sample(
    model: nn.Module,
    diffusion: CrystalDiffusion,
    state: Data,
    timestep: int,
) -> Data:
    """Take one DDPM reverse step from ``x_t`` to ``x_{t-1}``."""
    if not 0 <= timestep < diffusion.betas.numel():
        raise ValueError("timestep is outside the diffusion schedule")

    output = model(state, torch.tensor([timestep], device=state.lattice.device))
    epsilon_lattice = _prediction(output, "lattice")
    epsilon_coords = _prediction(output, "frac_coords")
    epsilon_atom = _prediction(output, "atom_latent")
    beta = diffusion.betas[timestep]
    alpha = 1.0 - beta
    alpha_bar = diffusion.sqrt_alpha_bars[timestep].square()
    alpha_bar_previous = (
        diffusion.sqrt_alpha_bars[timestep - 1].square()
        if timestep > 0
        else torch.ones_like(alpha_bar)
    )
    posterior_variance = beta * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
    mean_scale = beta / diffusion.sqrt_one_minus_alpha_bars[timestep]
    result = state.clone()
    result.lattice = (
        state.lattice - mean_scale * epsilon_lattice
    ) / alpha.sqrt()
    result.frac_coords = torch.remainder(
        (state.frac_coords - mean_scale * epsilon_coords) / alpha.sqrt(),
        1.0,
    )
    result.atom_latent = (
        state.atom_latent - mean_scale * epsilon_atom
    ) / alpha.sqrt()
    result.x = result.atom_latent
    if timestep > 0:
        standard_noise = torch.randn_like(result.lattice)
        result.lattice += posterior_variance.sqrt() * standard_noise
        result.frac_coords = torch.remainder(
            result.frac_coords + posterior_variance.sqrt() * torch.randn_like(result.frac_coords),
            1.0,
        )
        result.atom_latent += posterior_variance.sqrt() * torch.randn_like(
            result.atom_latent
        )
        result.x = result.atom_latent
    return result


@torch.no_grad()
def generate(
    model: nn.Module,
    diffusion: CrystalDiffusion,
    num_atoms: int,
    device: str | torch.device = "cpu",
) -> Data:
    """Generate one crystal by denoising Gaussian noise from ``T-1`` to zero."""
    if num_atoms < 1:
        raise ValueError("num_atoms must be positive")
    device = torch.device(device)
    model.eval()
    diffusion.eval()
    dtype = diffusion.atom_embedding.weight.dtype
    state = Data(
        lattice=_initial_lattice(device, dtype),
        frac_coords=torch.remainder(
            torch.randn((num_atoms, 3), device=device, dtype=dtype), 1.0
        ),
        atom_types=torch.zeros(num_atoms, device=device, dtype=torch.long),
        atom_latent=torch.randn(
            (num_atoms, diffusion.atom_embedding.embedding_dim),
            device=device,
            dtype=dtype,
        ),
    )
    state.x = state.atom_latent
    for timestep in range(diffusion.betas.numel() - 1, -1, -1):
        state = p_sample(model, diffusion, state, timestep)
        state.frac_coords = torch.remainder(state.frac_coords, 1.0)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-atoms", type=int, required=True)
    parser.parse_args()
    parser.error(
        "construct the trained CrystalGNN and CrystalDiffusion, then call generate(...)"
    )


if __name__ == "__main__":
    main()
