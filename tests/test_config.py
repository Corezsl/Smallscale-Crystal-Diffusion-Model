from crystal_diffusion.config import load_config


def test_load_config_reads_defaults() -> None:
    config = load_config()

    assert isinstance(config, dict)
    assert config["data"]["cutoff"] == 5.0
    assert config["training"]["device"] == "cpu"
    assert config["materials_project"]["api_key_env"] == "MP_API_KEY"
