"""Download low-energy inorganic structures from the Materials Project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mp_api.client import MPRester


def download_structures(
    api_key: str,
    output_path: Path,
    count: int = 1_000,
    energy_above_hull: float = 0.05,
) -> int:
    """Download exactly ``count`` structures satisfying the energy criterion."""
    if count < 1:
        raise ValueError("count must be positive")
    if energy_above_hull < 0:
        raise ValueError("energy_above_hull must be non-negative")

    fields = ["material_id", "structure", "energy_above_hull"]
    with MPRester(api_key) as rester:
        summaries = rester.materials.summary.search(
            energy_above_hull=(0, energy_above_hull),
            fields=fields,
            num_chunks=2,
            chunk_size=count,
        )

    records: list[dict[str, Any]] = []
    for summary in summaries:
        if summary.energy_above_hull >= energy_above_hull:
            continue
        structure = summary.structure
        if structure is None:
            continue
        records.append(
            {
                "material_id": str(summary.material_id),
                "energy_above_hull": float(summary.energy_above_hull),
                "structure": structure.as_dict(),
            }
        )
        if len(records) == count:
            break

    if len(records) != count:
        raise RuntimeError(
            f"Materials Project returned only {len(records)} qualifying structures; "
            f"{count} were requested"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2)
        output_file.write("\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/materials_project.json"),
    )
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--energy-above-hull", type=float, default=0.05)
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        parser.error("set the MP_API_KEY environment variable before downloading")

    downloaded = download_structures(
        api_key=api_key,
        output_path=args.output,
        count=args.count,
        energy_above_hull=args.energy_above_hull,
    )
    print(f"Downloaded {downloaded} structures to {args.output}")


if __name__ == "__main__":
    main()
