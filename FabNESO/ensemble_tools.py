"""A module to create input directories and encode configurations for FabNESO."""

from __future__ import annotations

import itertools
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


def create_dir_tree(
    *,
    sweep_path: Path,
    n_dirs: int,
    destructive: bool,
    copy_dir: Path,
    edit_file: str,
    parameter_to_scan: str,
    scan_range: tuple[float, float],
    outdir_prefix: str,
) -> None:
    """Create a directory tree in the sweep_path."""
    copy_dir = Path(copy_dir)
    if not copy_dir.is_dir():
        msg = f"copy_dir {copy_dir} does not exist"
        raise FileNotFoundError(msg)
    if not (copy_dir / edit_file).is_file():
        msg = f"edit_file {copy_dir / edit_file} does not exist"
        raise FileNotFoundError(msg)
    if parameter_to_scan is None:
        msg = "parameter_to_scan not defined"
        raise TypeError(msg)
    if parameter_to_scan == "":
        msg = "parameter_to_scan left empty"
        raise ValueError(msg)

    # Make the base directory
    if sweep_path.is_dir():
        if destructive:
            shutil.rmtree(sweep_path)
        else:
            msg = f"Path {sweep_path} already exists and not in destructive mode"
            raise FileExistsError(msg)
    sweep_path.mkdir(parents=True)

    for i in range(n_dirs):
        para_val = calculate_parameter_value(n_dirs, scan_range[0], scan_range[1], i)
        new_dir = Path(sweep_path) / "SWEEP" / f"{outdir_prefix}{i}"
        shutil.copytree(copy_dir, new_dir)
        # Now we edit the parameter file for our
        # template scan if we're doing that
        edit_parameters(new_dir / edit_file, {parameter_to_scan: para_val})


def calculate_parameter_value(
    n_dirs: int,
    initial_value: float,
    final_value: float,
    iteration: int,
) -> float:
    """Return the value of the parameter at a given iteration."""
    return (
        initial_value
        if n_dirs == 1
        else initial_value + (iteration / (n_dirs - 1)) * (final_value - initial_value)
    )


def _product_dict(input_dict: dict) -> Iterator[dict]:
    """Compute a Cartesian product of a dictionary of iterables."""
    keys = input_dict.keys()
    for values in itertools.product(*input_dict.values()):
        yield dict(zip(keys, values, strict=True))


def indices_iterator(n_dirs: int, n_parameters: int) -> Iterator[tuple[Any, ...]]:
    """Create an iterator for the indices of the dictionary sweep."""
    yield from itertools.product(*(range(n_dirs),) * n_parameters)


def create_dict_sweep(
    *,
    sweep_path: Path,
    n_dirs: int,
    destructive: bool,
    copy_dir: Path,
    edit_file: str,
    parameter_dict: dict[str, tuple[float, float]],
) -> None:
    """Use a dictionary with each parameter interval to create a sweep directory."""
    # If destructive, delete the whole tree if it already exists
    if destructive and sweep_path.is_dir():
        shutil.rmtree(sweep_path)
    # Uniformly spaced grids on [low, high] for each parameter
    parameter_grids = {
        key: [calculate_parameter_value(n_dirs, low, high, i) for i in range(n_dirs)]
        for key, (low, high) in parameter_dict.items()
    }
    # Compute Cartesian products of all parameter value combinations plus grid indices
    for parameter_values, indices in zip(
        _product_dict(parameter_grids),
        indices_iterator(n_dirs, len(parameter_grids)),
        strict=True,
    ):
        directory_name = return_directory_name(parameter_values, indices)
        directory_path = Path(sweep_path) / "SWEEP" / directory_name
        shutil.copytree(copy_dir, directory_path)
        edit_parameters(directory_path / edit_file, parameter_values)


def return_directory_name(parameter_values: dict, indices: tuple[Any, ...]) -> str:
    """Return the directory name given parameter names and indices."""
    return "-".join(f"{k}_{i}" for k, i in zip(parameter_values, indices, strict=True))


def list_parameter_values(conditions_file: Path, parameter_name: str) -> list[str]:
    """Return a list of the values of a given parameter_name in conditions_file."""
    data = ElementTree.parse(conditions_file)  # noqa: S314
    root = data.getroot()
    conditions = root.find("CONDITIONS")
    if conditions is None:
        msg = f"Failed to find CONDITIONS in the file {conditions_file}"
        raise ValueError(msg)
    parameters = conditions.find("PARAMETERS")
    if parameters is None:
        msg = (
            "Failed to find PARAMETERS in the CONDITIONS node" f" of {conditions_file}"
        )
        raise ValueError(msg)

    # List of matched parameter values
    values = []
    for element in parameters.iter("P"):
        match = re.match(
            r"\s*(?P<key>\w+)\s*=\s*(?P<value>-?\d*(\.?\d)+)\s*", str(element.text)
        )
        if match is None:
            msg = f"Parameter definition of unexpected format: {element.text}"
            raise ValueError(msg)
        key = match.group("key")
        value = str(match.group("value"))
        if key == parameter_name:
            values.append(value)
    return values


def edit_parameters(
    conditions_file: Path, parameter_overrides: Mapping[str, float | str]
) -> None:
    """Edit parameters in the configuration file to the desired value."""
    parser = ElementTree.XMLParser(  # noqa: S314
        target=ElementTree.TreeBuilder(insert_comments=True)
    )
    data = ElementTree.parse(conditions_file, parser=parser)  # noqa: S314
    root = data.getroot()
    conditions = root.find("CONDITIONS")
    if conditions is None:
        msg = f"Conditions file {conditions_file} does not contain a CONDITIONS node."
        raise ValueError(msg)
    parameters = conditions.find("PARAMETERS")
    if parameters is None:
        msg = f"Conditions file {conditions_file} does not contain a PARAMETERS node."
        raise ValueError(msg)
    for element in parameters.iter("P"):
        if element.text is None:
            msg = f"Parameter element {element} does contain a definition."
            raise ValueError(msg)
        match = re.match(r"\s*(?P<key>\w+)\s*=", element.text)
        if match is None:
            msg = f"Parameter definition of unexpected format: {element.text}"
            raise ValueError(msg)
        key = match.group("key")
        if key in parameter_overrides:
            element.text = f" {key} = {parameter_overrides[key]} "
    data.write(conditions_file)
