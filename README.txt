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