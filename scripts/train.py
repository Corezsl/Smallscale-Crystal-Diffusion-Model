"""Train a CrystalGNN to predict forward diffusion noise."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.loader import DataLoader

from crystal_diffusion.diffusion import CrystalDiffusion

LOGGER = logging.getLogger("crystal_diffusion.train")
NOISE_FIELDS = ("lattice", "frac_coords", "atom_latent")


def _prediction(prediction: Mapping[str, Tensor] | Any, field: str) -> Tensor:
    if isinstance(prediction, Mapping):
        value = prediction.get(field)
    else:
        value = getattr(prediction, field, None)
    if value is None:
        raise ValueError(f"model output is missing noise prediction: {field}")
    return value


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    diffusion: CrystalDiffusion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one epoch and return the mean element-wise MSE loss."""
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        num_graphs = batch.num_graphs
        timesteps = torch.randint(
            0, diffusion.betas.numel(), (num_graphs,), device=device
        )
        noise = {
            "lattice": torch.randn_like(batch.lattice.float()),
            "frac_coords": torch.randn_like(batch.frac_coords.float()),
            "atom_latent": torch.randn(
                (batch.atom_types.numel(), diffusion.atom_embedding.embedding_dim),
                device=device,
                dtype=batch.frac_coords.dtype,
            ),
        }
        noisy_batch, target_noise = diffusion.q_sample(
            batch, timesteps, noise=noise, return_noise=True
        )
        prediction = model(noisy_batch, timesteps)
        loss = sum(
            nn.functional.mse_loss(_prediction(prediction, field), target_noise[field])
            for field in NOISE_FIELDS
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach())
        batches += 1
    if batches == 0:
        raise ValueError("cannot train on an empty DataLoader")
    return total_loss / batches


def train(
    model: nn.Module,
    loader: DataLoader,
    diffusion: CrystalDiffusion,
    epochs: int,
    learning_rate: float = 1e-4,
    device: str | torch.device = "cpu",
) -> list[float]:
    """Train ``model`` with AdamW and return the loss for each epoch."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = torch.device(device)
    model.to(device)
    diffusion.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(diffusion.parameters()),
        lr=learning_rate,
    )
    losses = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, diffusion, optimizer, device)
        losses.append(loss)
        LOGGER.info("epoch %d/%d - loss %.6f", epoch, epochs, loss)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error(
        "construct a CrystalGNN and CrystalDataset, then call train(model, loader, diffusion, ...)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
