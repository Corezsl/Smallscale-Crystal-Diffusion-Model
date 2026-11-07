import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure

from crystal_diffusion.data.dataset import CrystalDataset


def test_dataset_preserves_periodic_neighbor_images() -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    data = CrystalDataset([structure])[0]

    assert data.lattice.shape == (3, 3)
    assert data.frac_coords.shape == (1, 3)
    assert data.atom_types.tolist() == [14]
    assert data.edge_index.shape[0] == 2
    assert data.edge_image.shape[1] == 3
    assert torch.all(data.edge_distances <= 5.0 + 1e-5)
    assert torch.any(torch.any(data.edge_image != 0, dim=1))
