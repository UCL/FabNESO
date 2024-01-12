"""Functions for reading the outputs of the NESO solvers."""

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np


def read_hdf5_datasets(
    hdf5_file_path: Path,
    dataset_paths: dict[str, str | Sequence[str]],
) -> dict[str, np.ndarray]:
    """Read HDF5 dataset output from NESO."""
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        return {
            key: np.asarray(hdf5_file[path_or_paths])
            if isinstance(path_or_paths, str)
            else np.stack([hdf5_file[path] for path in path_or_paths])
            for key, path_or_paths in dataset_paths.items()
        }
