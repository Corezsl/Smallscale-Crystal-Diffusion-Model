import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from crystal_diffusion.diffusion import CrystalDiffusion


def test_q_sample_wraps_coordinates_and_diffuses_atom_latents() -> None:
    data = Data(
        lattice=torch.eye(3),
        frac_coords=torch.tensor([[0.2, 0.5, 0.9]]),
        atom_types=torch.tensor([14]),
    )
    diffusion = CrystalDiffusion(timesteps=10, atom_latent_dim=8)

    noisy = diffusion.q_sample(data, t=9)

    assert noisy.atom_latent.shape == (1, 8)
    assert noisy.x.shape == (1, 8)
    assert torch.all(noisy.frac_coords >= 0.0)
    assert torch.all(noisy.frac_coords < 1.0)
    assert not torch.equal(noisy.atom_latent, diffusion.atom_embedding(data.atom_types))


def test_q_sample_rejects_invalid_timestep() -> None:
    diffusion = CrystalDiffusion(timesteps=10)
    data = Data(
        lattice=torch.eye(3),
        frac_coords=torch.zeros((1, 3)),
        atom_types=torch.tensor([1]),
    )

    with pytest.raises(ValueError, match="timestep"):
        diffusion.q_sample(data, t=10)
