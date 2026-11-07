"""Configuration utilities for repository defaults and workflow settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = {
    "data": {"cutoff": 5.0, "batch_size": 32},
    "model": {
        "timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "atom_latent_dim": 64,
        "num_elements": 118,
    },
    "training": {"epochs": 100, "learning_rate": 1e-4, "device": "cpu"},
    "materials_project": {
        "api_key_env": "MP_API_KEY",
        "max_structures": 1000,
        "energy_above_hull_threshold": 0.05,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config() -> dict[str, Any]:
    """Return the built-in default configuration."""
    return dict(_DEFAULT_CONFIG)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config data and merge with the repository defaults."""
    config_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise TypeError("Configuration file must contain a dictionary at the top level")

    return _deep_merge(_DEFAULT_CONFIG, raw)
