"""Tests for the ensemble_tools utilities."""

import shutil
from functools import reduce
from operator import mul
from pathlib import Path
from typing import get_args

import pytest

from FabNESO.ensemble_tools import (
    SamplingRule,
    _grid_directory_name,
    _indices_iterator,
    _qmc_directory_name,
    _uniformly_spaced_samples,
    create_grid_ensemble,
    create_qmc_ensemble,
    edit_parameters,
    list_parameter_values,
)

CONFIG_FILES_PATH = Path(__file__).parents[1] / "config_files"


def test_edit_parameters(tmp_path: Path) -> None:
    """Test the edit_parameters method of ensemble_tools."""
    parameter_to_test = "particle_initial_velocity"
    parameter_test_value = 10.0
    temp_conditions_path = tmp_path / "conditions.xml"
    shutil.copyfile(
        CONFIG_FILES_PATH / "two_stream" / "conditions.xml",
        temp_conditions_path,
    )

    edit_parameters(temp_conditions_path, {parameter_to_test: parameter_test_value})

    n_equal_in_value, n_different_in_value = _check_parameter_in_conditions(
        temp_conditions_path, parameter_to_test, parameter_test_value
    )
    assert n_equal_in_value == 1
    assert n_different_in_value == 0


def _check_parameter_in_conditions(
    conditions_file_name: Path, parameter_name: str, expected_value: float
) -> tuple[int, int]:
    """Return number of matched and failed parameters in conditions_file_name."""
    values = list_parameter_values(conditions_file_name, parameter_name)
    # Number of matched and different parameters
    n_equals = 0
    n_different = 0
    for value in values:
        if expected_value == pytest.approx(float(value)):
            n_equals += 1
        else:
            n_different += 1

    return (n_equals, n_different)


def test_list_parameter_values() -> None:
    """Test the list_parameter_values method in ensemble_tools."""
    conditions_file = CONFIG_FILES_PATH / "two_stream" / "conditions.xml"
    parameter_values = list_parameter_values(
        conditions_file, "particle_initial_velocity"
    )
    # Check that we only find one instance
    assert len(parameter_values) == 1
    # Check the default parameter value
    original_value = 1.0
    assert float(parameter_values[0]) == original_value
    # Test a fake parameter
    parameter_values = list_parameter_values(conditions_file, "fake_parameter_not_real")
    assert len(parameter_values) == 0
    # Test raise a ValueError when using an incorrect xml file
    with pytest.raises(ValueError, match=r".*Failed to find CONDITIONS.*"):
        parameter_values = list_parameter_values(
            CONFIG_FILES_PATH / "two_stream" / "mesh.xml",
            "particle_initial_velocity",
        )


def test_check_parameter_in_conditions(tmp_path: Path) -> None:
    """Test the private xml parser method of this class."""
    parameter_to_test = "particle_initial_velocity"
    parameter_test_value = 10.0
    test_cond_path = tmp_path / "conditions.xml"
    shutil.copyfile(
        CONFIG_FILES_PATH / "two_stream" / "conditions.xml",
        test_cond_path,
    )
    edit_parameters(test_cond_path, {parameter_to_test: parameter_test_value})
    # Test that a different value of the test_value returns exactly 1 non-matched result
    n_equal_in_value, n_different_in_value = _check_parameter_in_conditions(
        test_cond_path, parameter_to_test, parameter_test_value + 10.0
    )
    assert n_equal_in_value == 0
    assert n_different_in_value == 1
    # Pass a fake parameter name. Should find 0 results
    n_equal_in_value, n_different_in_value = _check_parameter_in_conditions(
        test_cond_path, "fake_parameter", parameter_test_value
    )
    assert n_equal_in_value == 0
    assert n_different_in_value == 0


@pytest.mark.parametrize(
    "parameter_ranges",
    [
        {
            "particle_initial_velocity": (0.1, 2.5, 3),
            "particle_charge_density": (102.0, 108.0, 5),
        },
        {
            "particle_number_density": (102.0, 108.0, 5),
        },
    ],
)
def test_create_grid_ensemble(tmp_path: Path, parameter_ranges: dict) -> None:
    """Test the create_dict_sweep method of ensemble_tools."""
    output_path = tmp_path / "test" / "SWEEP"
    source_path = CONFIG_FILES_PATH / "two_stream"
    conditions_file = "conditions.xml"
    create_grid_ensemble(
        output_path=output_path,
        source_path=source_path,
        conditions_file=conditions_file,
        parameter_ranges=parameter_ranges,
    )
    grid_shape = tuple(n for *_, n in parameter_ranges.values())
    grid_size = reduce(mul, grid_shape)
    # Check we make the corect number of directories
    assert len(list(output_path.iterdir())) == grid_size

    parameter_values = {
        parameter_name: _uniformly_spaced_samples(*lower_upper_n_sample)
        for parameter_name, lower_upper_n_sample in parameter_ranges.items()
    }

    # Loop through the directories and check the conditions file
    for indices in _indices_iterator(grid_shape):
        directory_name = _grid_directory_name(list(parameter_ranges.keys()), indices)
        sample_directory_path = output_path / directory_name

        # Check the directory exists
        assert sample_directory_path.is_dir()

        # Check that all files have been copied correctly
        for f in source_path.iterdir():
            assert (sample_directory_path / f.name).is_file()

        # Check that the parameters have been edited correctly
        for i, parameter_name in enumerate(parameter_ranges.keys()):
            expected_value = parameter_values[parameter_name][indices[i]]
            n_equal_in_value, n_different_in_value = _check_parameter_in_conditions(
                sample_directory_path / "conditions.xml", parameter_name, expected_value
            )
            assert n_equal_in_value == 1
            assert n_different_in_value == 0


@pytest.mark.parametrize("n_sample", [1, 5, 10])
@pytest.mark.parametrize("seed", [1234, 42])
@pytest.mark.parametrize("rule", get_args(SamplingRule))
@pytest.mark.parametrize(
    "parameter_intervals",
    [
        {
            "particle_initial_velocity": (0.1, 2.5),
            "particle_charge_density": (80.0, 120.0),
        },
        {
            "particle_number_density": (90.0, 110.0),
        },
    ],
)
def test_create_qmc_ensemble(
    tmp_path: Path, n_sample: int, seed: int, rule: str, parameter_intervals: dict
) -> None:
    """Test the create_dict_sweep method of ensemble_tools."""
    output_path = tmp_path / "test" / "SWEEP"
    source_path = CONFIG_FILES_PATH / "two_stream"
    conditions_file = "conditions.xml"
    create_qmc_ensemble(
        output_path=output_path,
        source_path=source_path,
        conditions_file=conditions_file,
        n_sample=n_sample,
        seed=seed,
        rule=rule,
        parameter_intervals=parameter_intervals,
    )
    # Check we make the corect number of directories
    assert len(list(output_path.iterdir())) == n_sample

    # Loop through the directories and check the conditions file
    for sample_index in range(n_sample):
        directory_name = _qmc_directory_name(
            list(parameter_intervals.keys()), sample_index
        )
        sample_directory_path = output_path / directory_name

        # Check the directory exists
        assert sample_directory_path.is_dir()

        # Check that all files have been copied correctly
        for f in source_path.iterdir():
            assert (sample_directory_path / f.name).is_file()


@pytest.mark.parametrize("n_sample", [1, 3, 100])
@pytest.mark.parametrize(
    "lower_upper",
    [
        (1.0, 3.0),
        (10000, 20000),
        (5.0, 4.0),
    ],
)
def test_uniformly_spaced_samples(
    lower_upper: tuple[float, float], n_sample: int
) -> None:
    """Tests the _uniformly_spaced_samples method of ensemble_tools."""
    lower, upper = lower_upper
    parameter_values = _uniformly_spaced_samples(lower, upper, n_sample)
    # Check the returned list has the correct number of entries
    assert len(parameter_values) == n_sample
    # Check the values in the generated list are unique
    n_unique_entries = len(set(parameter_values))
    assert len(parameter_values) == n_unique_entries
    for value in parameter_values:
        assert lower <= value <= upper


@pytest.mark.parametrize(
    "grid_shape",
    [
        [1, 3, 100],
        [20, 20, 20],
        [50, 2, 1],
    ],
)
@pytest.mark.parametrize(
    "parameter_list",
    [
        ["particle_initial_velocity", "particle_charge_density"],
        [
            "particle_initial_velocity",
            "particle_charge_density",
            "particle_number_density",
        ],
    ],
)
def test_grid_directory_name(grid_shape: list[int], parameter_list: list[str]) -> None:
    """Test the _grid_directory_name function."""
    directory_names = []
    # Create a dummy set of indices based on n_dirs
    for indices in _indices_iterator(grid_shape[: len(parameter_list)]):
        dir_name = _grid_directory_name(parameter_list, indices)
        # Check that each parameter only appears once in the directory name
        for parameter in parameter_list:
            assert dir_name.count(parameter) == 1
        directory_names.append(dir_name)
    # Check we've made the correct number of directories
    assert len(directory_names) == reduce(mul, grid_shape[: len(parameter_list)])
    # Check that we've made unique directories
    n_unique_dirs = len(set(directory_names))
    assert len(directory_names) == n_unique_dirs


@pytest.mark.parametrize(
    "grid_shape",
    [
        [1, 3, 7],
        [2, 5, 10],
        [1],
        [4, 6, 9, 1, 5, 4],
    ],
)
def test_indices_iterator(grid_shape: list[int]) -> None:
    """Test the indices_iterator from the ensemble_tools."""
    indices_list = []
    for indices in _indices_iterator(grid_shape):
        assert len(indices) == len(grid_shape)
        indices_list.append(indices)
    assert len(indices_list) == reduce(mul, grid_shape)
    n_unique_indices = len(set(indices_list))
    assert n_unique_indices == len(indices_list)
