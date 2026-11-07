"""Command-line entry points for common crystal-diffusion tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from crystal_diffusion.config import load_config


def _print_usage_example(name: str, example: str) -> None:
    print(f"{name}: {example}")


def download_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small set of Materials Project structures.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/materials_project.json"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--energy-above-hull", type=float, default=0.05)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.count > config["materials_project"]["max_structures"]:
        print(
            "Requested count exceeds the default maximum in config.yaml; "
            "the command may still work, but you may want to adjust the setting."
        )
    _print_usage_example(
        "download",
        "python scripts/download_mp.py --count 10 --output data/processed/materials_project.json",
    )
    print(f"Loaded configuration from {config}")


def train_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Guidance entry point for crystal diffusion training.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Training configuration: epochs={args.epochs or config['training']['epochs']}")
    print("This project exposes training as a Python API; construct a CrystalGNN and DataLoader in your own code.")
    print("Example: train(model, loader, diffusion, epochs=10, learning_rate=1e-4)")


def generate_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Guidance entry point for crystal generation.",
    )
    parser.add_argument("--num-atoms", type=int, default=8)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    _ = load_config(args.config)
    print(f"Generate a crystal with {args.num_atoms} atoms.")
    print("This project provides generation utilities, but the trained CrystalGNN still needs to be supplied by your pipeline.")


def evaluate_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Guidance entry point for crystal evaluation.",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    _ = load_config(args.config)
    if args.input is None:
        print("Provide a structure path or a generated output in your own evaluation script.")
    else:
        print(f"Would evaluate structure file: {args.input}")
    print("For a full evaluation, use evaluate_structure(...) from scripts/evaluate.py or a custom script.")
