"""Crystal data structures and dataset utilities."""

from __future__ import annotations

from .dataset import CrystalDataset
from .synthetic import make_synthetic_crystal_dataset

__all__ = ["CrystalDataset", "make_synthetic_crystal_dataset"]
