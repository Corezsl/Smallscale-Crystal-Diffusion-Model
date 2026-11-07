"""Forward diffusion for periodic crystal structures."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data


class CrystalDiffusion(nn.Module):
    """Apply a linear-schedule forward process to a crystal graph.

    Lattice matrices use ordinary Gaussian diffusion. Fractional coordinates
    use the same process followed by modulo-one wrapping. Atomic numbers are
    looked up in a learned embedding table and the resulting vectors are
    diffused as continuous variables in ``Data.atom_latent``.
    """

    def __init__(
        self,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        atom_latent_dim: int = 64,
        num_elements: int = 118,
    ) -> None:
        super().__init__()
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        if atom_latent_dim < 1:
            raise ValueError("atom_latent_dim must be positive")
        if num_elements < 1:
            raise ValueError("num_elements must be positive")

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.atom_embedding = nn.Embedding(num_elements + 1, atom_latent_dim)

    def _schedule_value(self, values: Tensor, timestep: int | Tensor) -> Tensor:
        timestep_tensor = torch.as_tensor(timestep, device=values.device)
        if timestep_tensor.numel() != 1:
            raise ValueError("q_sample currently accepts one timestep per crystal")
        timestep_index = int(timestep_tensor.item())
        if not 0 <= timestep_index < len(values):
            raise ValueError(f"timestep must be in [0, {len(values) - 1}]")
        return values[timestep_index]

    def q_sample(self, x_0: Data, t: int | Tensor) -> Data:
        """Return ``x_t`` sampled from a crystal graph ``x_0`` at timestep ``t``."""
        required_fields = ("lattice", "frac_coords", "atom_types")
        missing_fields = [field for field in required_fields if not hasattr(x_0, field)]
        if missing_fields:
            raise ValueError(f"crystal data is missing fields: {missing_fields}")

        alpha = self._schedule_value(self.sqrt_alpha_bars, t)
        sigma = self._schedule_value(self.sqrt_one_minus_alpha_bars, t)
        result = x_0.clone()

        lattice = x_0.lattice.float()
        coords = x_0.frac_coords.float()
        lattice_noise = torch.randn_like(lattice)
        coordinate_noise = torch.randn_like(coords)
        result.lattice = alpha * lattice + sigma * lattice_noise
        result.frac_coords = torch.remainder(
            alpha * coords + sigma * coordinate_noise, 1.0
        )

        atom_types = x_0.atom_types
        if atom_types.dtype.is_floating_point:
            raise TypeError("x_0.atom_types must contain integer atomic numbers")
        if torch.any(atom_types < 0) or torch.any(
            atom_types >= self.atom_embedding.num_embeddings
        ):
            raise ValueError("atomic numbers exceed the configured embedding table")
        atom_latent = self.atom_embedding(atom_types.long())
        result.atom_latent = (
            alpha * atom_latent + sigma * torch.randn_like(atom_latent)
        )
        result.x = result.atom_latent
        return result

    def forward(self, x_0: Data, t: int | Tensor) -> Data:
        return self.q_sample(x_0, t)
