"""Streamlit interface for crystal diffusion generation and evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _structure_from_upload(uploaded_file: Any):
    from pymatgen.core import Structure

    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".json", ".mson"}:
        return Structure.from_dict(json.loads(content))
    fmt = "cif" if suffix == ".cif" else None
    return Structure.from_str(content.decode("utf-8"), fmt=fmt)


def _show_structure(structure: Any) -> None:
    st.write(f"**Formula:** {structure.composition.reduced_formula}")
    st.write(f"**Sites:** {len(structure)}")
    st.write(f"**Volume:** {structure.volume:.3f} Å³")
    st.dataframe(
        [
            {
                "site": index,
                "element": str(site.specie),
                "a": float(site.frac_coords[0]),
                "b": float(site.frac_coords[1]),
                "c": float(site.frac_coords[2]),
            }
            for index, site in enumerate(structure.sites)
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.line_chart(
        {
            "fractional a": [float(site.frac_coords[0]) for site in structure.sites],
            "fractional b": [float(site.frac_coords[1]) for site in structure.sites],
            "fractional c": [float(site.frac_coords[2]) for site in structure.sites],
        }
    )


def _evaluation_tab() -> None:
    st.header("Evaluate a crystal")
    uploaded_file = st.file_uploader(
        "Upload a CIF or Pymatgen JSON structure",
        type=["cif", "json", "mson"],
        key="evaluation_upload",
    )
    minimum_distance = st.number_input(
        "Minimum periodic atom distance (Å)",
        min_value=0.1,
        value=0.5,
        step=0.1,
    )
    if uploaded_file is None:
        st.info("Upload a structure to run the validity and EMT checks.")
        return
    try:
        structure = _structure_from_upload(uploaded_file)
        _show_structure(structure)
        from evaluate import evaluate_structure

        result = evaluate_structure(structure, minimum_distance)
        if result.valid:
            st.success("No periodic atom overlap was detected.")
        else:
            st.error(f"Atoms are too close: {result.minimum_distance:.4f} Å")
        if result.energy is None:
            st.warning(result.energy_error or "EMT energy was unavailable.")
        else:
            st.metric("ASE EMT energy", f"{result.energy:.4f} eV")
    except (OSError, ValueError, TypeError) as error:
        st.error(f"Could not evaluate the uploaded structure: {error}")


def _generation_tab() -> None:
    st.header("Generate a crystal")
    st.number_input("Number of atoms", min_value=1, value=8, step=1, key="num_atoms")
    st.number_input("Diffusion steps", min_value=1, value=1_000, step=100, key="steps")
    checkpoint = st.file_uploader(
        "Optional serialized trained model",
        type=["pt", "pth", "ckpt"],
        key="model_upload",
        help="A CrystalGNN implementation is not included yet; provide a serialized nn.Module.",
    )
    if st.button("Generate", type="primary"):
        if checkpoint is None:
            st.warning(
                "Generation requires a trained CrystalGNN checkpoint. "
                "The repository currently provides the sampler but not the GNN architecture."
            )
            return
        st.error(
            "Checkpoint loading is intentionally not automatic. Instantiate the matching "
            "CrystalGNN architecture before loading its state dictionary."
        )


def _training_tab() -> None:
    st.header("Training configuration")
    epochs = st.number_input("Epochs", min_value=1, value=100, step=10)
    batch_size = st.number_input("Batch size", min_value=1, value=32, step=1)
    learning_rate = st.number_input(
        "Learning rate", min_value=1e-8, value=1e-4, format="%.2e"
    )
    dataset_path = st.text_input(
        "Dataset path",
        value="data/processed/materials_project.json",
    )
    st.code(
        "python scripts/train.py  # call train(model, loader, diffusion, ...)",
        language="bash",
    )
    st.caption(
        f"Configured for {epochs} epochs, batch size {batch_size}, "
        f"learning rate {learning_rate:.2e}. Dataset: {dataset_path}"
    )
    st.info(
        "Training is exposed as a programmatic API until the CrystalGNN architecture "
        "and dataset-loading command are finalized."
    )


def main() -> None:
    st.set_page_config(
        page_title="Crystal Diffusion Studio",
        page_icon="◇",
        layout="wide",
    )
    st.title("Crystal Diffusion Studio")
    st.caption("Explore, evaluate, and configure periodic crystal generation.")
    generation, evaluation, visualization, training = st.tabs(
        ["Generation", "Evaluation", "Visualization", "Training"]
    )
    with generation:
        _generation_tab()
    with evaluation:
        _evaluation_tab()
    with visualization:
        st.header("Visualization")
        uploaded_file = st.file_uploader(
            "Upload a CIF or Pymatgen JSON structure",
            type=["cif", "json", "mson"],
            key="visualization_upload",
        )
        if uploaded_file is None:
            st.info("Upload a structure to inspect its fractional coordinates.")
        else:
            try:
                _show_structure(_structure_from_upload(uploaded_file))
            except (OSError, ValueError, TypeError) as error:
                st.error(f"Could not read the uploaded structure: {error}")
    with training:
        _training_tab()


if __name__ == "__main__":
    main()
