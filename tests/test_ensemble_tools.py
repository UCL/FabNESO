"""Tests for the ensemble_tools utilities."""

import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from FabNESO.ensemble_tools import (
    calculate_parameter_value,
    create_dict_sweep,
    create_dir_tree,
    edit_parameters,
    indices_iterator,
    list_parameter_values,
    return_directory_name,
)


def test_edit_parameters(tmp_path: Path) -> None:
    """Test the edit_parameters method of ensemble_tools."""
    parameter_to_test = "particle_initial_velocity"
    parameter_test_value = 10.0
    temp_conditions_path = tmp_path / "conditions.xml"
    shutil.copyfile(
        Path(__file__).parents[1] / "config_files" / "two_stream" / "conditions.xml",
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
    conditions_file = (
        Path(__file__).parents[1] / "config_files" / "two_stream" / "conditions.xml"
    )
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
            Path(__file__).parents[1] / "config_files" / "two_stream" / "mesh.xml",
            "particle_initial_velocity",
        )


def test_check_parameter_in_conditions(tmp_path: Path) -> None:
    """Test the private xml parser method of this class."""
    parameter_to_test = "particle_initial_velocity"
    parameter_test_value = 10.0
    test_cond_path = tmp_path / "conditions.xml"
    shutil.copyfile(
        Path(__file__).parents[1] / "config_files" / "two_stream" / "conditions.xml",
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


@pytest.mark.parametrize("n_dirs", [1, 3, 5, 10])
@pytest.mark.parametrize("destructive", [True, False])
@pytest.mark.parametrize(
    "parameter_and_range",
    [
        ["particle_initial_velocity", -4.0, 5.0],
        ["particle_charge_density", 101.0, 108.0],
        ["particle_number_density", 101.0, 108.0],
    ],
)
def test_create_dir_tree(
    tmp_path: Path,
    *,
    n_dirs: int,
    destructive: bool,
    parameter_and_range: list,
) -> None:
    """Test the create_dir_tree method of ensemble_tools."""
    test_sweep_dir = tmp_path / "test_sweep"
    copy_dir = Path("config_files") / "two_stream"
    edit_file = "conditions.xml"
    parameter_to_scan = parameter_and_range[0]
    scan_range = (parameter_and_range[1], parameter_and_range[2])
    outdir_prefix = "asd"

    create_dir_tree(
        sweep_path=test_sweep_dir,
        n_dirs=n_dirs,
        destructive=destructive,
        copy_dir=copy_dir,
        edit_file=edit_file,
        parameter_to_scan=parameter_to_scan,
        scan_range=scan_range,
        outdir_prefix=outdir_prefix,
    )

    assert (test_sweep_dir / "SWEEP").is_dir()
    assert len(list((test_sweep_dir / "SWEEP").iterdir())) == n_dirs
    for i in range(n_dirs):
        cond_file = test_sweep_dir / "SWEEP" / f"{outdir_prefix}{i}" / "conditions.xml"
        assert cond_file.is_file()
        para_value = calculate_parameter_value(n_dirs, scan_range[0], scan_range[1], i)
        # Check the parameter has been edited correctly and appears only once
        n_equal_in_value, n_different_in_value = _check_parameter_in_conditions(
            cond_file, parameter_to_scan, para_value
        )
        assert n_equal_in_value == 1
        assert n_different_in_value == 0


class _CreateDirTreeArgsDict(TypedDict):
    sweep_path: Path
    n_dirs: int
    destructive: bool
    copy_dir: Path
    edit_file: str
    parameter_to_scan: str
    scan_range: tuple[float, float]
    outdir_prefix: str


def test_exceptions_create_dir_tree(tmp_path: Path) -> None:
    """Test exceptions thrown in create_dir_path."""
    argument_dict = _CreateDirTreeArgsDict(
        sweep_path=tmp_path / "test_sweep",
        n_dirs=5,
        destructive=True,
        copy_dir=Path("config_files") / "two_stream",
        edit_file="conditions.xml",
        parameter_to_scan="particle_initial_velocity",
        scan_range=(0.2, 5.0),
        outdir_prefix="asd",
    )
    # Should raise an exception when trying to edit the mesh file
    argument_dict["edit_file"] = "mesh.xml"
    with pytest.raises(ValueError, match=r".* a CONDITIONS node.*"):
        create_dir_tree(**argument_dict)
    # The directory already exists, so running without destructive should
    # raise a FileExistsError
    argument_dict["destructive"] = False
    argument_dict["edit_file"] = "conditions.xml"
    with pytest.raises(
        FileExistsError, match=".*ready exists and not in destructive .*"
    ):
        create_dir_tree(**argument_dict)
    # fake_name.xml doesn't exist, should raise FileNotFoundError
    argument_dict["destructive"] = True
    argument_dict["edit_file"] = "fake_name.xml"
    with pytest.raises(FileNotFoundError, match=".*does not exist"):
        create_dir_tree(**argument_dict)
    argument_dict["edit_file"] = "conditions.xml"
    argument_dict["parameter_to_scan"] = ""
    with pytest.raises(ValueError, match=".*left empty"):
        create_dir_tree(**argument_dict)


@pytest.mark.parametrize(
    "n_dirs",
    [
        [1, 3, 5],
    ],
)
@pytest.mark.parametrize("destructive", [True, False])
@pytest.mark.parametrize(
    "parameter_dict",
    [
        {
            "particle_initial_velocity": (0.1, 2.5),
            "particle_charge_density": (102.0, 108.0),
        },
        {
            "num_particles_total": (20000, 1000000),
            "particle_number_density": (102.0, 108.0),
        },
    ],
)
def test_create_dict_sweep(
    tmp_path: Path, *, n_dirs: list[int], destructive: bool, parameter_dict: dict
) -> None:
    """Test the create_dict_sweep method of ensemble_tools."""
    sweep_path = tmp_path / "test"
    copy_dir = Path("config_files") / "two_stream"
    edit_file = "conditions.xml"
    # Combine the n_dirs vector into the parameter_dict
    useable_n_dirs = n_dirs[: len(parameter_dict)]
    for (key, item), n_dir in zip(parameter_dict.items(), useable_n_dirs, strict=True):
        parameter_dict[key] = item[:2] + (n_dir,)

    create_dict_sweep(
        sweep_path=sweep_path,
        destructive=destructive,
        copy_dir=copy_dir,
        edit_file=edit_file,
        parameter_dict=parameter_dict,
    )
    n_total_directories = np.prod(n_dirs[: len(parameter_dict)])
    # Check we make the corect number of directories
    assert len(list((sweep_path / "SWEEP").iterdir())) == n_total_directories

    # Loop through the directories and check the conditions file
    for indices in indices_iterator(n_dirs[: len(parameter_dict)]):
        directory_name = return_directory_name(list(parameter_dict.keys()), indices)
        this_dir = sweep_path / "SWEEP" / directory_name

        # Check the directory exists
        assert this_dir.is_dir()

        # Check that all files have been copied correctly
        for f in copy_dir.iterdir():
            assert (this_dir / f.name).is_file()

        # Check that the parameters have been edited correctly
        for i in range(len(parameter_dict)):
            parameter = list(parameter_dict.keys())[i]
            scan_range = parameter_dict[parameter]
            para_value = calculate_parameter_value(
                scan_range[2], scan_range[0], scan_range[1], indices[i]
            )
            n_equal_in_value, n_different_in_value = _check_parameter_in_conditions(
                this_dir / "conditions.xml", parameter, para_value
            )
            assert n_equal_in_value == 1
            assert n_different_in_value == 0


@pytest.mark.parametrize("n_dirs", [1, 3, 100])
@pytest.mark.parametrize(
    "scan_range",
    [
        [1.0, 3.0],
        [10000, 20000],
        [5.0, 4.0],
    ],
)
def test_calculate_parameter_value(n_dirs: int, scan_range: list[float]) -> None:
    """Tests the calculate_parameter_value method of ensemble_tools."""
    parameter_values = [
        calculate_parameter_value(n_dirs, scan_range[0], scan_range[1], i)
        for i in range(n_dirs)
    ]
    # Check the returned list has the correct number of entries
    assert len(parameter_values) == n_dirs
    # Check the values in the generated list are unique
    n_unique_entries = len(set(parameter_values))
    assert len(parameter_values) == n_unique_entries
    for value in parameter_values:
        assert min(scan_range) <= value <= max(scan_range)


@pytest.mark.parametrize(
    "n_dirs",
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
def test_return_directory_name(n_dirs: list[int], parameter_list: list[str]) -> None:
    """Test the return_directory_name function."""
    directory_names = []
    # Create a dummy set of indices based on n_dirs
    for indices in indices_iterator(n_dirs[: len(parameter_list)]):
        dir_name = return_directory_name(parameter_list, indices)
        # Check that each parameter only appears once in the directory name
        for parameter in parameter_list:
            assert dir_name.count(parameter) == 1
        directory_names.append(dir_name)
    # Check we've made the correct number of directories
    assert len(directory_names) == np.prod(n_dirs[: len(parameter_list)])
    # Check that we've made unique directories
    n_unique_dirs = len(set(directory_names))
    assert len(directory_names) == n_unique_dirs


@pytest.mark.parametrize(
    "n_dirs",
    [
        [1, 3, 7],
        [2, 5, 10],
        [1],
        [4, 6, 9, 1, 5, 4],
    ],
)
def test_indices_iterator(n_dirs: list[int]) -> None:
    """Test the indices_iterator from the ensemble_tools."""
    indices_list = []
    for indices in indices_iterator(n_dirs):
        assert len(indices) == len(n_dirs)
        indices_list.append(indices)
    assert len(indices_list) == np.prod(n_dirs)
    n_unique_indices = len(set(indices_list))
    assert n_unique_indices == len(indices_list)
