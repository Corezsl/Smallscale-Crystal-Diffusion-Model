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