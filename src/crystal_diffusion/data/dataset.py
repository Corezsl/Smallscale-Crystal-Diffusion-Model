"""PyTorch Geometric representation of periodic crystal structures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, Dataset


class CrystalDataset(Dataset):
    """Convert Pymatgen structures into periodic crystal graphs.

    ``Data.lattice`` stores the row-vector lattice matrix in Angstrom,
    ``Data.frac_coords`` stores fractional coordinates, and
    ``Data.atom_types`` stores atomic numbers. ``Data.edge_image`` stores the
    integer lattice translation applied to the destination atom for each edge.
    """

    def __init__(
        self,
        structures: Sequence[Structure],
        cutoff: float = 5.0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.structures = tuple(structures)
        if not all(isinstance(structure, Structure) for structure in self.structures):
            raise TypeError("structures must contain only Pymatgen Structure objects")
        self.cutoff = float(cutoff)
        super().__init__(
            root=None,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def len(self) -> int:
        return len(self.structures)

    def get(self, idx: int) -> Data:
        structure = self.structures[idx]
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float32)
        frac_coords = np.asarray(structure.frac_coords, dtype=np.float32)

        try:
            atomic_numbers = [site.specie.Z for site in structure.sites]
        except (AttributeError, TypeError) as error:
            raise ValueError(
                "CrystalDataset requires ordered structures with one element per site"
            ) from error

        center, neighbor, offsets, _ = structure.get_neighbor_list(self.cutoff)
        edge_index = torch.as_tensor(
            np.stack((center, neighbor), axis=0), dtype=torch.long
        )
        edge_image = torch.as_tensor(offsets, dtype=torch.long)

        fractional_displacement = (
            frac_coords[neighbor] + offsets - frac_coords[center]
        )
        cartesian_displacement = fractional_displacement @ lattice
        edge_distances = np.linalg.norm(cartesian_displacement, axis=1).astype(
            np.float32
        )

        return Data(
            x=torch.as_tensor(atomic_numbers, dtype=torch.long).view(-1, 1),
            atom_types=torch.as_tensor(atomic_numbers, dtype=torch.long),
            frac_coords=torch.from_numpy(frac_coords),
            lattice=torch.from_numpy(lattice),
            edge_index=edge_index,
            edge_image=edge_image,
            edge_attr=torch.from_numpy(edge_distances).view(-1, 1),
            edge_distances=torch.from_numpy(edge_distances),
            num_nodes=len(structure),
        )
