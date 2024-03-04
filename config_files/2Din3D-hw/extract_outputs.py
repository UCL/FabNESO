"""Extract energy and enstrophy values from CSV file and output to JSON file."""

import argparse
import json
from pathlib import Path

import pandas as pd


def extract_outputs(results_directory: str) -> dict[str, list[float]]:
    """Extract energy and enstrophy values from CSV file."""
    growth_rates_dataframe = pd.read_csv(Path(results_directory) / "growth_rates.csv")
    return {col: growth_rates_dataframe[col].to_list() for col in ("E", "W")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_directory",
        type=Path,
        help="Directory solver outputs were written to",
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to write JSON output file to"
    )
    args = parser.parse_args()
    outputs = extract_outputs(args.results_directory)
    with args.output_file.open("w") as f:
        json.dump(outputs, f)
