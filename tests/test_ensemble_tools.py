"""Tests for the ensemble_tools utilities."""

import itertools
import shutil
from pathlib import Path
from typing import TypedDict

import pytest

from FabNESO.ensemble_tools import (
    calculate_parameter_value,
    create_dict_sweep,
    create_dir_tree,
    edit_parameters,
    list_parameter_values,
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


@pytest.mark.parametrize("n_dirs", [1, 3, 5, 10])
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
    tmp_path: Path, *, n_dirs: int, destructive: bool, parameter_dict: dict
) -> None:
    """Test the create_dict_sweep method of ensemble_tools."""
    sweep_path = tmp_path / "test"
    copy_dir = Path("config_files") / "two_stream"
    edit_file = "conditions.xml"
    create_dict_sweep(
        sweep_path=sweep_path,
        n_dirs=n_dirs,
        destructive=destructive,
        copy_dir=copy_dir,
        edit_file=edit_file,
        parameter_dict=parameter_dict,
    )
    n_total_directories = n_dirs ** len(parameter_dict)
    # Check we make the corect number of directories
    assert len(list((sweep_path / "SWEEP").iterdir())) == n_total_directories

    # Loop through the directories and check the conditions file
    for indices in itertools.product(*(range(n_dirs),) * len(parameter_dict)):
        directory_name = "-".join(
            f"{k}_{i}" for k, i in zip(parameter_dict, indices, strict=True)
        )
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
                n_dirs, scan_range[0], scan_range[1], indices[i]
            )
            n_equal_in_value, n_different_in_value = _check_parameter_in_conditions(
                this_dir / "conditions.xml", parameter, para_value
            )
            assert n_equal_in_value == 1
            assert n_different_in_value == 0
