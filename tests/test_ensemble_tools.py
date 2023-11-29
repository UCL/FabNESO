""" Test the utility scripts """

import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import itertools

import pytest

# from plugins.FabNESO.utils.make_sweep_dir
# from ensemble_tools import edit_parameters
from utils.ensemble_tools import edit_parameters, create_dir_tree, create_dict_sweep


def test_edit_parameters(tmpdir):
    """Test the edit_parameters method"""

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


def check_parameter_in_conditions(conditions_file_name, parameter_name, expected_value):
    """Returns True if parameter_pattern is found in the PARAMETERS secion"""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    data = ET.parse(conditions_file_name, parser=parser)
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


def test_create_dir_tree(tmpdir):
    """Test the create_dir_tree() method"""

    test_sweep_dir = Path(tmpdir) / "test_sweep"
    n_dirs = 5
    destructive = True
    copy_dir = Path("config_files") / "two_stream"
    edit_file = "conditions.xml"
    parameter_to_scan = "particle_initial_velocity"
    scan_range = (0.2, 5.0)
    outdir_prefix = "asd"

    create_dir_tree(
        test_sweep_dir,
        n_dirs,
        destructive,
        copy_dir,
        edit_file,
        parameter_to_scan,
        scan_range,
        outdir_prefix,
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
    with pytest.raises(ValueError):
        create_dir_tree(
            test_sweep_dir,
            n_dirs,
            destructive,
            copy_dir,
            edit_file,
            parameter_to_scan,
            scan_range,
            outdir_prefix,
        )


def test_create_dict_sweep(tmpdir):
    """Test the create_dict_sweep method of utils/ensemble_tools.py"""
    sweep_path = Path(tmpdir) / "test"
    n_divs = 5
    destructive = True
    copy_dir = Path("config_files") / "two_stream"
    edit_file = "conditions.xml"
    parameter_dict = {
        "particle_initial_velocity": [0.1, 0.9],
        "particle_charge_density": [102, 108],
    }
    create_dict_sweep(
        sweep_path, n_divs, destructive, copy_dir, edit_file, parameter_dict
    )
    n_total_directories = n_divs ** len(parameter_dict)
    # Check we make the corect number of directories
    assert len(list((sweep_path / "SWEEP").iterdir())) == n_total_directories

    # Loop through the directories and check the conditions file
    for indices in itertools.product(*(range(n_divs),) * len(parameter_dict)):
        directory_name = "-".join(f"{k}_{i}" for k, i in zip(parameter_dict, indices))
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
            with pytest.raises(ValueError):
                # The mesh file should be not edited
                assert check_parameter_in_conditions(
                    this_dir / "mesh.xml", parameter, para_value
                )
