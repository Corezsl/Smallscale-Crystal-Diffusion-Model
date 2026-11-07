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
        if timestep_tensor.numel() == 0 or timestep_tensor.ndim > 1:
            raise ValueError("timestep must be a scalar or one-dimensional tensor")
        if torch.any((timestep_tensor < 0) | (timestep_tensor >= len(values))):
            raise ValueError(f"timestep must be in [0, {len(values) - 1}]")
        return values[timestep_tensor.long()]

    def q_sample(
        self,
        x_0: Data,
        t: int | Tensor,
        noise: dict[str, Tensor] | None = None,
        return_noise: bool = False,
    ) -> Data | tuple[Data, dict[str, Tensor]]:
        """Return ``x_t`` sampled from ``x_0`` at timestep ``t``.

        A one-dimensional ``t`` tensor supplies one timestep per graph in a
        PyG batch. Pass ``return_noise=True`` to retrieve the exact Gaussian
        targets used to construct the sample.
        """
        required_fields = ("lattice", "frac_coords", "atom_types")
        missing_fields = [field for field in required_fields if not hasattr(x_0, field)]
        if missing_fields:
            raise ValueError(f"crystal data is missing fields: {missing_fields}")

        timestep = torch.as_tensor(t, device=x_0.lattice.device)
        alpha = self._schedule_value(self.sqrt_alpha_bars, timestep)
        sigma = self._schedule_value(self.sqrt_one_minus_alpha_bars, timestep)
        if timestep.numel() > 1:
            if not hasattr(x_0, "batch"):
                raise ValueError("per-graph timesteps require a batched PyG Data object")
            num_graphs = int(x_0.batch.max().item()) + 1
            if timestep.numel() != num_graphs:
                raise ValueError("number of timesteps must equal the number of graphs")
            lattice_alpha = alpha.view(-1, 1, 1)[x_0.batch.new_tensor(range(num_graphs))]
            lattice_sigma = sigma.view(-1, 1, 1)[
                x_0.batch.new_tensor(range(num_graphs))
            ]
            node_alpha = alpha[x_0.batch]
            node_sigma = sigma[x_0.batch]
        else:
            lattice_alpha = alpha
            lattice_sigma = sigma
            node_alpha = alpha
            node_sigma = sigma
        result = x_0.clone()

        lattice = x_0.lattice.float()
        coords = x_0.frac_coords.float()
        atom_types = x_0.atom_types
        if noise is None:
            noise = {
                "lattice": torch.randn_like(lattice),
                "frac_coords": torch.randn_like(coords),
                "atom_latent": torch.randn(
                    (x_0.atom_types.shape[0], self.atom_embedding.embedding_dim),
                    device=coords.device,
                    dtype=self.atom_embedding.weight.dtype,
                ),
            }
        expected_noise = {"lattice", "frac_coords", "atom_latent"}
        if set(noise) != expected_noise:
            raise ValueError(f"noise must contain exactly {sorted(expected_noise)}")
        lattice_noise = noise["lattice"]
        coordinate_noise = noise["frac_coords"]
        atom_noise = noise["atom_latent"]
        if lattice_noise.shape != lattice.shape or coordinate_noise.shape != coords.shape:
            raise ValueError("lattice and coordinate noise shapes must match x_0")
        if atom_noise.shape != (
            atom_types.shape[0],
            self.atom_embedding.embedding_dim,
        ):
            raise ValueError("atom latent noise has an invalid shape")
        result.lattice = lattice_alpha * lattice + lattice_sigma * lattice_noise
        result.frac_coords = torch.remainder(
            node_alpha * coords + node_sigma * coordinate_noise, 1.0
        )

        if atom_types.dtype.is_floating_point:
            raise TypeError("x_0.atom_types must contain integer atomic numbers")
        if torch.any(atom_types < 0) or torch.any(
            atom_types >= self.atom_embedding.num_embeddings
        ):
            raise ValueError("atomic numbers exceed the configured embedding table")
        atom_latent = self.atom_embedding(atom_types.long())
        result.atom_latent = (
            node_alpha * atom_latent + node_sigma * atom_noise
        )
        result.x = result.atom_latent
        if return_noise:
            return result, noise
        return result

    def forward(self, x_0: Data, t: int | Tensor) -> Data:
        return self.q_sample(x_0, t)
