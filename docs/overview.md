# Smallscale Crystal Diffusion Model

This directory contains a short overview of the project goals and scope.

## Project goal

The repository models diffusion processes over periodic crystal structures using:

- lattice matrices,
- fractional coordinates,
- atom-type latent vectors.

## Scope

The code is intentionally structured as a research scaffold. It provides the primitives needed to represent crystal graphs, apply forward diffusion, and evaluate generated structures, but it does not yet include a fully packaged, production-ready model architecture.

## Typical workflow

1. Load or synthesize crystal data.
2. Build a graph-based dataset.
3. Train a CrystalGNN to predict diffusion noise.
4. Sample from the reverse process.
5. Decode and validate generated structures.

## Best practices

- Keep your model architecture explicit and versioned.
- Store training runs and checkpoints separately.
- Validate generated structures with periodic-distance checks.
- Treat the synthetic dataset and Materials Project data as separate experiments.
