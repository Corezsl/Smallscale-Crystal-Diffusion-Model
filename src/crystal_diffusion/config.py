"""Configuration utilities for repository defaults and workflow settings."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Validated runtime configuration for the crystal diffusion project."""

    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    materials_project: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "data": copy.deepcopy(self.data),
            "model": copy.deepcopy(self.model),
            "training": copy.deepcopy(self.training),
            "materials_project": copy.deepcopy(self.materials_project),
        }


_DEFAULT_CONFIG = ProjectConfig(
    data={"cutoff": 5.0, "batch_size": 32},
    model={
        "timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "atom_latent_dim": 64,
        "num_elements": 118,
    },
    training={"epochs": 100, "learning_rate": 1e-4, "device": "cpu"},
    materials_project={
        "api_key_env": "MP_API_KEY",
        "max_structures": 1000,
        "energy_above_hull_threshold": 0.05,
    },
)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionary values while preserving default values."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def default_config() -> dict[str, Any]:
    """Return the built-in default configuration as a dictionary."""
    return _DEFAULT_CONFIG.to_dict()


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate required keys and basic numeric constraints."""
    required_sections = {"data", "model", "training", "materials_project"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing required sections: {sorted(missing)}")

    if config["data"].get("cutoff", 0) <= 0:
        raise ValueError("data.cutoff must be positive")
    if config["model"].get("timesteps", 0) < 1:
        raise ValueError("model.timesteps must be at least 1")
    if config["training"].get("epochs", 0) < 1:
        raise ValueError("training.epochs must be at least 1")
    if config["materials_project"].get("max_structures", 0) < 1:
        raise ValueError("materials_project.max_structures must be at least 1")
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config data and merge it with repository defaults."""
    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2] / "config.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise TypeError("Configuration file must contain a dictionary at the top level")

    merged = _merge_dicts(_DEFAULT_CONFIG.to_dict(), raw)
    return _validate_config(merged)
