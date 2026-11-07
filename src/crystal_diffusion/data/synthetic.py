"""Helper utilities for constructing synthetic crystal datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pymatgen.core import Structure

from crystal_diffusion.data.dataset import CrystalDataset


def make_synthetic_structure(
    num_atoms: int = 8,
    seed: int = 0,
    lattice_scale: float = 6.0,
) -> Structure:
    """Create a small crystal-like structure with random fractional coordinates."""
    if num_atoms < 1:
        raise ValueError("num_atoms must be positive")
    rng = np.random.default_rng(seed)
    lattice = np.eye(3, dtype=np.float32) * lattice_scale
    raw_coords = rng.random((num_atoms, 3), dtype=np.float32)
    atomic_numbers = [int(x) for x in rng.integers(1, 20, size=num_atoms)]
    return Structure(lattice, atomic_numbers, raw_coords)


def make_synthetic_crystal_dataset(
    num_structures: int = 4,
    num_atoms: int = 8,
    cutoff: float = 5.0,
    seed: int = 0,
) -> CrystalDataset:
    """Build a tiny synthetic dataset for demos and tests."""
    if num_structures < 1:
        raise ValueError("num_structures must be positive")
    structures = [
        make_synthetic_structure(num_atoms=num_atoms, seed=seed + index)
        for index in range(num_structures)
    ]
    return CrystalDataset(structures, cutoff=cutoff)
