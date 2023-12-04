"""A module to create input directories and encode configurations for FabNESO."""

from __future__ import annotations

import itertools
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
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

    # Set the initial value of the scanned parameter to the lower limit
    para_val = scan_range[0]

    for i in range(n_dirs):
        new_dir = Path(sweep_path) / "SWEEP" / f"{outdir_prefix}{i}"
        shutil.copytree(copy_dir, new_dir)
        # Now we edit the parameter file for our
        # template scan if we're doing that
        edit_parameters(new_dir / edit_file, {parameter_to_scan: para_val})
        # iterate para_val
        para_val += (
            0 if n_dirs == 1 else (scan_range[1] - scan_range[0]) / float(n_dirs - 1)
        )


def _product_dict(input_dict: dict) -> Iterator[dict]:
    """Compute a Cartesian product of a dictionary of iterables."""
    keys = input_dict.keys()
    for values in itertools.product(*input_dict.values()):
        yield dict(zip(keys, values, strict=True))


def create_dict_sweep(
    *,
    sweep_path: Path,
    n_divs: int,
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
    parameter_grids = (
        {key: [low] for key, (low, high) in parameter_dict.items()}
        if n_divs == 1
        else {
            key: [low + (i / (n_divs - 1)) * (high - low) for i in range(n_divs)]
            for key, (low, high) in parameter_dict.items()
        }
    )
    # Compute Cartesian products of all parameter value combinations plus grid indices
    for parameter_values, indices in zip(
        _product_dict(parameter_grids),
        itertools.product(*(range(n_divs),) * len(parameter_dict)),
        strict=True,
    ):
        directory_name = "-".join(
            f"{k}_{i}" for k, i in zip(parameter_values, indices, strict=True)
        )
        directory_path = Path(sweep_path) / "SWEEP" / directory_name
        shutil.copytree(copy_dir, directory_path)
        edit_parameters(directory_path / edit_file, parameter_values)


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
