""" A module to create input directories and encode
configfurations for FabNESO
"""

import os
import shutil
import itertools
import re
from pathlib import Path
import lxml.etree as ET


def create_dir_tree(
    sweep_path,
    n_dirs,
    destructive,
    copy_dir,
    edit_file,
    parameter_to_scan,
    scan_range,
    outdir_prefix,
):
    """Create a directory tree in the sweep_path"""

    copy_files = os.path.isdir(copy_dir)
    if not copy_files:
        print("No correct copy dir found, just creating the tree for now!")

    # Make the base directory
    try:
        make_directory(sweep_path, destructive)
    except FileExistsError:
        return

    # Set the initial value of the scanned parameter to the lower limit
    para_val = scan_range[0]

    for i in range(n_dirs):
        new_dir = Path(sweep_path) / "SWEEP" / outdir_prefix + i
        # Make the directory
        os.makedirs(new_dir)
        # If we're copying files, do so
        if copy_files:
            copy_dir_contents(new_dir, copy_dir)
        # Now we edit the parameter file for our
        # template scan if we're doing that
        if os.path.isfile(new_dir + edit_file) and parameter_to_scan:
            edit_parameter(new_dir + edit_file, parameter_to_scan, para_val)
        # iterate para_val
        para_val += (scan_range[1] - scan_range[0]) / float(n_dirs - 1)


def create_dict_sweep(
    sweep_path, n_dirs, destructive, copy_dir, edit_file, parameter_dict
):
    """Use a dictionary with each parameter's high and low to create
    a multi-dimensional sweep directory"""

    print(
        f"{len(parameter_dict)} parameters found, "
        f"requiring {n_dirs} divisions per parameter. "
        f"Creating {n_dirs ** len(parameter_dict)} configurations..."
    )

    # Calculate all of the parameter values and put them in a dict
    scan_points = {}
    list_all = []
    for parameter in parameter_dict.keys():
        param_value = parameter_dict[parameter][0]
        scan_points[parameter] = []
        for i in range(n_dirs):
            scan_points[parameter].append(param_value)
            list_all.append(f"{parameter}_{i}")
            param_value += (
                parameter_dict[parameter][1] - parameter_dict[parameter][0]
            ) / float(n_dirs - 1)

    # Create the combinations of these parameters.
    combinations = list(itertools.combinations(list_all, len(parameter_dict.keys())))
    all_configs = []
    for comb in combinations:
        record_comb = True
        for i in range(len(comb) - 1):
            search_term = comb[i][: len(comb[i]) - comb[i][::-1].find("_")]
            for j in range(i + 1, len(comb)):
                if comb[j].startswith(search_term):
                    record_comb = False
                    break
        if record_comb:
            all_configs.append(comb)

    # If destructive, let's delete the whole tree if it already exists
    if destructive and os.path.isdir(sweep_path):
        shutil.rmtree(sweep_path)
    # Check whether the copy directory exists
    copy_files = os.path.isdir(copy_dir)
    # Now loop over each combination, create a directory, copy the
    # configs and conditions, and edit the configuation to contain
    # the correct parameters.
    for comb in all_configs:
        dir_name = "-".join(comb)
        dir_path = Path(sweep_path) / "SWEEP" / dir_name
        make_directory(dir_path, destructive)
        if copy_files:
            copy_dir_contents(dir_path, copy_dir)
        # Create parameter map for conditions encoding
        parameters = {}
        for parameter in parameter_dict.keys():
            param_index = [item for item in comb if item.startswith(parameter)][
                0
            ].split("_")[-1]
            parameters[parameter] = scan_points[parameter][int(param_index)]
        if edit_file:
            encode_conditions_file(Path(dir_path) / edit_file, parameters)


def copy_dir_contents(dir_path, copy_dir):
    """Copy the contents of copy_dir -> dir_path"""

    shutil.copytree(copy_dir, dir_path, dirs_exist_ok=True)


def make_directory(directory_name, destructive):
    """Make the directory tree at directory_name."""

    if os.path.isdir(directory_name):
        if destructive:
            shutil.rmtree(directory_name)
        else:
            msg = "Path already exists and not in destructive mode"
            raise FileExistsError(msg)
    os.makedirs(directory_name)


def encode_conditions_file(in_file_name, param_dict):
    """Encode a configuration file with a dict of input values"""

    for param_name in param_dict.keys():
        edit_parameter(in_file_name, param_name, param_dict[param_name])


def edit_parameter(in_file_name, param, val):
    """Edit a single parameter in the configuration
    file to the desired value"""

    print(f"Edit {param} : {val}")

    data = ET.parse(in_file_name)
    root = data.getroot()
    parameters = root.find("CONDITIONS").find("PARAMETERS")
    for element in parameters.iter("P"):
        match = re.match(
            r"\s*(?P<key>\w+)\s*=\s*(?P<value>[0-9.-]+)\s*",
            element.text,
        )
        if match is None:
            # This happens for the constructed elements, so skip them
            continue
        key = match.group("key")
        if key == param:
            new_element = ET.Element("P")
            new_element.text = f" {param} = {val} "
            parameters.insert(parameters.index(element), new_element)
            parameters.remove(element)
    ET.indent(root, space="    ")
    data.write(in_file_name)
