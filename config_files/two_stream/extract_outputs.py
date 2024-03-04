""""Extract line field derivatives at final time from HDF5 file and output to JSON."""

import argparse
import json
from pathlib import Path

import h5py


def _get_step_keys(hdf5_file: h5py.File) -> list[str]:
    """Get ordered list of step keys."""
    return sorted(hdf5_file.keys(), key=lambda k: int(k.split("#")[-1]))


def extract_outputs(results_directory: str) -> dict[str, list[float]]:
    """Extract line field derivatives at final time from HDF5 file."""
    hdf5_file_path = (
        Path(results_directory)
        / "Electrostatic2D3V_line_field_deriv_evaluations.h5part"
    )
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        step_keys = _get_step_keys(hdf5_file)
        last_step_key = step_keys[-1]
        return {
            "x": list(hdf5_file[last_step_key]["x"]),
            "phi": list(hdf5_file[last_step_key]["FIELD_EVALUATION_0"]),
        }


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
