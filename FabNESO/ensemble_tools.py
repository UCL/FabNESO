"""A module to create input directories and encode configurations for FabNESO."""

from __future__ import annotations

import itertools
import re
import shutil
from typing import TYPE_CHECKING, Literal
from xml.etree import ElementTree

import chaospy

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path
    from typing import TypeAlias


def _uniformly_spaced_samples(
    lower: float,
    upper: float,
    n_sample: int,
) -> list[float]:
    """Return `n_sample` evenly spaced samples over an interval `[lower, upper]`."""
    return (
        [lower]
        if n_sample == 1
        else [lower + (i / (n_sample - 1)) * (upper - lower) for i in range(n_sample)]
    )


def _product_dict(input_dict: dict) -> Iterator[dict]:
    """Compute a Cartesian product of a dictionary of iterables."""
    keys = input_dict.keys()
    for values in itertools.product(*input_dict.values()):
        yield dict(zip(keys, values, strict=True))


def _indices_iterator(grid_shape: Iterable[int]) -> Iterator[tuple[int, ...]]:
    """Create an iterator for the indices of a grid."""
    yield from itertools.product(*[range(size) for size in grid_shape])


def _grid_directory_name(
    parameter_names: Iterable[str], indices: tuple[int, ...]
) -> str:
    """Return a directory name given parameter names and grid indices."""
    return "-".join(f"{k}_{i}" for k, i in zip(parameter_names, indices, strict=True))


def create_grid_ensemble(
    *,
    output_path: Path,
    source_path: Path,
    conditions_file: str,
    parameter_ranges: dict[str, tuple[float, float, int]],
) -> None:
    """
    Create an ensemble directory corresponding to parameters on tensor product grid.

    Args:
        output_path: Path to directory to write ensemble files to.
        source_path: Path to directory containing 'source' configuration to create
            ensemble from by varying parameters in conditions file.
        conditions_file: Name of conditions file in `source_path` directory.
        parameter_ranges: Dictionary mapping from parameter names to tuples specifying
            in order the lower bound, upper bound and number of samples in evenly spaced
            grids on each parameter to be varied, with overall grid being the tensor
            product of these per-parameter grids.

    """
    # Uniformly spaced grids on [low, high] for each parameter
    parameter_grids = {
        key: _uniformly_spaced_samples(*parameter_range)
        for key, parameter_range in parameter_ranges.items()
    }
    # Compute Cartesian products of all parameter value combinations plus grid indices
    for parameter_values, indices in zip(
        _product_dict(parameter_grids),
        _indices_iterator([n_sample for *_, n_sample in parameter_ranges.values()]),
        strict=True,
    ):
        directory_name = _grid_directory_name(list(parameter_values.keys()), indices)
        directory_path = output_path / directory_name
        shutil.copytree(source_path, directory_path)
        edit_parameters(directory_path / conditions_file, parameter_values)


def _qmc_directory_name(parameter_names: list[str], sample_index: int) -> str:
    """Return a directory name given parameter names and sample index."""
    return "-".join(parameter_names) + f"_{sample_index}"


SamplingRule: TypeAlias = Literal[
    "additive_recursion", "hammersley", "korobov", "latin_hypercube", "random", "sobol"
]


def create_qmc_ensemble(
    *,
    output_path: Path,
    source_path: Path,
    conditions_file: str,
    n_sample: int,
    seed: int,
    rule: str,
    parameter_intervals: dict[str, tuple[float, float]],
) -> None:
    """
    Create an ensemble directory corresponding to quasi-Monte Carlo parameter samples.

    Args:
        output_path: Path to directory to write ensemble files to.
        source_path: Path to directory containing 'source' configuration to create
            ensemble from by varying parameters in conditions file.
        conditions_file: Name of conditions file in `source_path` directory.
        parameter_intervals: Dictionary mapping from parameter names to tuples
            specifying in order the lower and upper bounds of uniform distribution on
            each each parameter to be varied, with overall joint distribution on
            parameters corresponding to the product of these distributions (that is
            assuming independence across the parameters.)

    """
    uniform_distribution = chaospy.J(
        *(
            chaospy.Uniform(lower, upper)
            for lower, upper in parameter_intervals.values()
        )
    )
    parameter_samples = uniform_distribution.sample(
        n_sample, seed=seed, rule=rule, include_axis_dim=True
    )
    for index, parameter_values in enumerate(parameter_samples.T):
        directory_name = _qmc_directory_name(list(parameter_intervals.keys()), index)
        directory_path = output_path / directory_name
        shutil.copytree(source_path, directory_path)
        parameter_overrides = dict(
            zip(parameter_intervals.keys(), parameter_values, strict=True)
        )
        edit_parameters(directory_path / conditions_file, parameter_overrides)


def list_parameter_values(conditions_file: Path, parameter_name: str) -> list[str]:
    """
    Return a list of the values of a given parameter name in conditions_file.

    Args:
        conditions_file: Path to conditions file to inspect.
        parameter_name: Name of parameter to get values for.

    Returns:
        List of values for parameter with name `parameter_name`.

    """
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
    conditions_file: Path,
    parameter_overrides: Mapping[str, float | str],
    *,
    create_missing: bool = False,
) -> None:
    """
    Edit parameters in a conditions file.

    Args:
        conditions_file: Source conditions file to edit parameters in.
        parameter_overrides: Mapping from parameter names to values to override the
            default value in the conditions file with.

    Keyword Args:
        create_missing: Whether to create new elements for parameters specified in
            `parameter_overrides` argument for which there is not an existing element
            in conditions file.

    """
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
    parameter_found = {para_name: False for para_name in parameter_overrides}
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
            parameter_found[key] = True
    if create_missing:
        for parameter in parameter_found:
            if not parameter_found[parameter]:
                new_para = ElementTree.Element("P")
                new_para.text = f" {parameter} = {parameter_overrides[parameter]} "
                parameters.append(new_para)
    data.write(conditions_file)
