Smallscale Crystal Diffusion Model
==================================

This project develops a continuous diffusion model for periodic crystal
structures. The model will jointly represent:

* a 3x3 lattice matrix in Euclidean space;
* fractional atomic coordinates on a unit torus; and
* categorical atom types.

Development setup
-----------------

Create an environment with Python 3.10 or newer, then install the project
dependencies:

    python -m pip install -r requirements.txt
    python -m pip install -e ".[dev]"

The source package is under ``src/crystal_diffusion``. The ``data``,
``diffusion``, and ``geometry`` packages are intentionally small at this
stage; their domain representations and model components will be added
incrementally.

Run the smoke tests with:

    python -m pytest

Download Materials Project structures
-------------------------------------

Set a Materials Project API key in ``MP_API_KEY`` and run:

    python scripts/download_mp.py

This writes up to 1,000 structures with ``energy_above_hull <= 0.05`` to
``data/processed/materials_project.json``. The downloader does not store the
API key in the repository.

Forward diffusion
-----------------

``crystal_diffusion.diffusion.CrystalDiffusion`` applies a linear beta
schedule. Lattice entries and atom-type embeddings receive Gaussian noise;
fractional coordinates are wrapped modulo one after noising.

Training
--------

The programmatic training entry point is ``scripts/train.py:train``. A
``CrystalGNN`` passed to it must accept ``(noisy_batch, timesteps)`` and return
noise predictions under the keys ``lattice``, ``frac_coords``, and
``atom_latent``. The loop samples one timestep per graph, uses AdamW, computes
the sum of the three MSE losses, and logs one loss per epoch.

Inference and evaluation
------------------------

``scripts/generate.py`` provides ``generate`` and ``p_sample`` for reverse
diffusion from Gaussian lattice, coordinate, and atom-latent noise. Coordinates
are wrapped modulo one after every reverse step. ``scripts/evaluate.py`` decodes
atom latents using nearest learned embeddings, constructs a Pymatgen
``Structure``, checks periodic close contacts, and attempts an ASE EMT energy
calculation. EMT may report an unsupported-element error for generated
compositions outside its element parameterization.

Interactive GUI
---------------

Install the optional Streamlit UI and launch it with:

    python -m pip install -e ".[app]"
    streamlit run scripts/app.py

The GUI supports CIF/Pymatgen JSON uploads, structure summaries, periodic
overlap checks, ASE EMT energy checks, generation configuration, and training
configuration. Generation becomes executable once a matching ``CrystalGNN``
architecture and checkpoint loader are supplied.