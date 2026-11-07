"""Convert generated tensors to Pymatgen structures and run sanity checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from ase import Atoms
from ase.calculators.emt import EMT
from pymatgen.core import Structure
from torch import Tensor

from crystal_diffusion.diffusion import CrystalDiffusion


@dataclass(frozen=True)
class EvaluationResult:
    """Basic physical and calculator checks for one generated structure."""

    structure: Structure
    atomic_numbers: list[int]
    valid: bool
    minimum_distance: float | None
    energy: float | None
    energy_error: str | None


def decode_atom_types(atom_latent: Tensor, diffusion: CrystalDiffusion) -> list[int]:
    """Decode continuous atom latents to the nearest learned element embedding."""
    embeddings = diffusion.atom_embedding.weight[1:]
    nearest = torch.cdist(atom_latent.float(), embeddings.float()).argmin(dim=1) + 1
    return nearest.detach().cpu().tolist()


def structure_from_tensors(
    lattice: Tensor,
    frac_coords: Tensor,
    atom_latent: Tensor,
    diffusion: CrystalDiffusion,
) -> Structure:
    atomic_numbers = decode_atom_types(atom_latent, diffusion)
    return Structure(
        lattice.detach().cpu().numpy(),
        atomic_numbers,
        torch.remainder(frac_coords, 1.0).detach().cpu().numpy(),
    )


def _minimum_periodic_distance(
    structure: Structure,
    cutoff: float,
) -> float | None:
    center, neighbor, offsets, distances = structure.get_neighbor_list(cutoff)
    valid_distances = [
        float(distance)
        for source, target, image, distance in zip(
            center, neighbor, offsets, distances
        )
        if source != target or any(image)
    ]
    return min(valid_distances) if valid_distances else None


def evaluate_structure(
    structure: Structure,
    minimum_distance: float = 0.5,
) -> EvaluationResult:
    """Check periodic overlaps and calculate an ASE EMT energy."""
    if minimum_distance <= 0:
        raise ValueError("minimum_distance must be positive")
    nearest_distance = _minimum_periodic_distance(
        structure, minimum_distance + 1e-6
    )
    valid = nearest_distance is None or nearest_distance >= minimum_distance
    energy: float | None = None
    energy_error: str | None = None
    try:
        atoms = Atoms(
            numbers=[int(site.specie.Z) for site in structure.sites],
            scaled_positions=structure.frac_coords,
            cell=structure.lattice.matrix,
            pbc=True,
        )
        atoms.calc = EMT()
        energy = float(atoms.get_potential_energy())
    except (KeyError, NotImplementedError, RuntimeError, ValueError) as error:
        energy_error = f"EMT energy unavailable: {error}"
    return EvaluationResult(
        structure=structure,
        atomic_numbers=[int(site.specie.Z) for site in structure.sites],
        valid=valid,
        minimum_distance=nearest_distance,
        energy=energy,
        energy_error=energy_error,
    )


def evaluate_generated(
    lattice: Tensor,
    frac_coords: Tensor,
    atom_latent: Tensor,
    diffusion: CrystalDiffusion,
    minimum_distance: float = 0.5,
) -> EvaluationResult:
    """Convert generated tensors and evaluate the resulting structure."""
    structure = structure_from_tensors(
        lattice, frac_coords, atom_latent, diffusion
    )
    return evaluate_structure(structure, minimum_distance)
