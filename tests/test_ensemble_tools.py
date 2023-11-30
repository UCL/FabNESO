"""Tests for the ensemble_tools utilities."""

import itertools
import re
import shutil
from pathlib import Path

import pytest
from defusedxml import ElementTree
from py.path import local

# from plugins.FabNESO.utils.make_sweep_dir
from FabNESO.ensemble_tools import create_dict_sweep, create_dir_tree, edit_parameters


def test_edit_parameters(tmpdir: local) -> None:
    """Test the edit_parameters method of ensemble_tools."""
    parameter_to_test = "particle_initial_velocity"
    parameter_test_value = 10.0
    temp_conditions_path = Path(tmpdir) / "conditions.xml"
    shutil.copyfile(
        Path("config_files") / "two_stream" / "conditions.xml", temp_conditions_path
    )

    edit_parameters(temp_conditions_path, {parameter_to_test: parameter_test_value})

    assert check_parameter_in_conditions(
        temp_conditions_path, parameter_to_test, parameter_test_value
    )


def check_parameter_in_conditions(
    conditions_file_name: Path, parameter_name: str, expected_value: float
) -> bool:
    """Return True if parameter_name has approx expected_value."""
    data = ElementTree.parse(conditions_file_name)
    root = data.getroot()
    conditions = root.find("CONDITIONS")
    if conditions is None:
        msg = f"Failed to find CONDITIONS in the file {conditions_file_name}"
        raise ValueError(msg)
    parameters = conditions.find("PARAMETERS")
    if parameters is None:
        msg = f"Failed to find PARAMETERS in the \
        CONDITIONS node of {conditions_file_name}"
        raise ValueError(msg)

    for element in parameters.iter("P"):
        match = re.match(
            r"\s*(?P<key>\w+)\s*=\s*(?P<value>-?\d*(\.?\d)+)\s*", element.text
        )
        if match is None:
            msg = f"Parameter definition of unexpected format: {element.text}"
            raise ValueError(msg)
        key = match.group("key")
        value = float(match.group("value"))
        if key == parameter_name:
            return expected_value == pytest.approx(value)  # Because of rounding errors

    # If we make it this far then we haven't found the parameter we were looking for,
    # which is surely a problem in its own right
    msg = f"Could not find the parameter {parameter_name} in {conditions_file_name}"
    raise ValueError(msg)


def test_create_dir_tree(tmpdir: local) -> None:
    """Test the create_dir_tree method of ensemble_tools."""
    test_sweep_dir = Path(tmpdir) / "test_sweep"
    n_dirs = 5
    destructive = True
    copy_dir = Path("config_files") / "two_stream"
    edit_file = "conditions.xml"
    parameter_to_scan = "particle_initial_velocity"
    scan_range = (0.2, 5.0)
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
        cond_file = test_sweep_dir / "SWEEP" / f"asd{i}" / "conditions.xml"
        assert cond_file.is_file()
        para_value = scan_range[0] + (i / (n_dirs - 1)) * (
            scan_range[1] - scan_range[0]
        )
        assert check_parameter_in_conditions(cond_file, parameter_to_scan, para_value)
    # Should raise an exception when trying to edit the mesh file
    edit_file = "mesh.xml"
    with pytest.raises(ValueError, match=r".* a CONDITIONS node.*"):
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


def test_create_dict_sweep(tmpdir: local) -> None:
    """Test the create_dict_sweep method of ensemble_tools."""
    sweep_path = Path(tmpdir) / "test"
    n_divs = 5
    destructive = True
    copy_dir = Path("config_files") / "two_stream"
    edit_file = "conditions.xml"
    parameter_dict = {
        "particle_initial_velocity": (0.1, 2.5),
        "particle_charge_density": (102.0, 108.0),
    }
    create_dict_sweep(
        sweep_path=sweep_path,
        n_divs=n_divs,
        destructive=destructive,
        copy_dir=copy_dir,
        edit_file=edit_file,
        parameter_dict=parameter_dict,
    )
    n_total_directories = n_divs ** len(parameter_dict)
    # Check we make the corect number of directories
    assert len(list((sweep_path / "SWEEP").iterdir())) == n_total_directories

    # Loop through the directories and check the conditions file
    for indices in itertools.product(*(range(n_divs),) * len(parameter_dict)):
        directory_name = "-".join(
            f"{k}_{i}" for k, i in zip(parameter_dict, indices, strict=True)
        )
        this_dir = sweep_path / "SWEEP" / directory_name

        # Check the directory exists
        assert this_dir.is_dir()

        # Check the conditions and mesh files exist
        assert (this_dir / "conditions.xml").is_file()
        assert (this_dir / "mesh.xml").is_file()

        # Check that the parameters have been edited correctly
        for i in range(len(parameter_dict)):
            parameter = list(parameter_dict.keys())[i]
            scan_range = parameter_dict[parameter]
            para_value = scan_range[0] + (indices[i] / (n_divs - 1)) * (
                scan_range[1] - scan_range[0]
            )
            assert check_parameter_in_conditions(
                this_dir / "conditions.xml", parameter, para_value
            )
            with pytest.raises(ValueError, match=r".*CONDITIONS.*"):
                # The mesh file should be not edited
                assert check_parameter_in_conditions(
                    this_dir / "mesh.xml", parameter, para_value
                )
