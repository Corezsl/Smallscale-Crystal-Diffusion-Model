from crystal_diffusion.data import make_synthetic_crystal_dataset


def test_synthetic_dataset_has_expected_size() -> None:
    dataset = make_synthetic_crystal_dataset(num_structures=3, num_atoms=4, cutoff=5.0)

    assert len(dataset) == 3
