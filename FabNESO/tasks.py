"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pyvbmc

try:
    from fabsim.base import fab
    from fabsim.deploy.templates import template

    fab.add_local_paths("FabNESO")
    FAB_IMPORTED = True
except ImportError:
    # If FabSim not installed we create a dummy fab namespace definining placeholder
    # decorators to make module still importable, for example when building docs
    from types import SimpleNamespace

    fab = SimpleNamespace(task=lambda f: f, load_plugin_env_vars=lambda _: lambda f: f)
    FAB_IMPORTED = False


from .ensemble_tools import create_dict_sweep, edit_parameters, list_parameter_values
from .read_outputs import read_hdf5_datasets


def _check_fab_module_imported() -> None:
    if not FAB_IMPORTED:
        msg = "fabsim.base.fab could not be imported - check FabSim3 is installed"
        raise ImportError(msg)


def _try_convert_to_int_and_check_positive(value: str | int, name: str) -> int:
    try:
        value = int(value)
    except ValueError as e:
        msg = f"{name} is not a valid integer literal: {value}"
        raise ValueError(msg) from e
    if value <= 0:
        msg = f"{name} must be a positive integer: {value}"
        raise ValueError(msg)
    return value


def _check_and_process_resource_args(
    processes: str | int,
    nodes: str | int,
    cpus_per_process: str | int,
    wall_time: str,
) -> tuple[int, int, int, str]:
    processes = _try_convert_to_int_and_check_positive(processes, "processes")
    nodes = _try_convert_to_int_and_check_positive(nodes, "nodes")
    cpus_per_process = _try_convert_to_int_and_check_positive(
        cpus_per_process, "cpus_per_process"
    )
    if processes % nodes != 0:
        msg = "processes must be a multiple of nodes"
        raise ValueError(msg)
    wall_time = wall_time.strip()
    match = re.match(r"^\d{1,2}:(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2})$", wall_time)
    if match is None:
        msg = "wall_time should be of format hh:mm:ss"
        raise ValueError(msg)
    minutes_in_hour = seconds_in_minute = 60
    if (
        int(match.group("minutes")) >= minutes_in_hour
        or int(match.group("seconds")) >= seconds_in_minute
    ):
        msg = "wall_time minute and second components should be in range 0 to 59"
        raise ValueError(msg)
    return processes, nodes, cpus_per_process, wall_time


def _create_job_args_dict(
    solver: str,
    conditions_file_name: str,
    mesh_file_name: str,
    processes: int,
    nodes: int,
    cpus_per_process: int,
    wall_time: str,
) -> dict[str, int | str]:
    return {
        "script": "neso",
        "neso_solver": solver,
        "neso_conditions_file": conditions_file_name,
        "neso_mesh_file": mesh_file_name,
        # FabSim convention is to use 'cores' to set number of MPI processes
        "cores": processes,
        "nodes": nodes,
        "corespernode": processes // nodes,
        "cpuspertask": cpus_per_process,
        "job_wall_time": wall_time,
    }


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    processes: str | int = 4,
    nodes: str | int = 1,
    cpus_per_process: str | int = 1,
    wall_time: str = "00:15:00",
    *,
    create_missing_parameters: bool = False,
    **parameter_overrides: str,
) -> None:
    """
    Run a single NESO solver instance.

    Args:
        config: Directory with single run configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        processes: Number of processes to run.
        nodes: Number of nodes to run on. Only applicable when running on a multi-node
            system.
        cpus_per_process: Number of processing units to use per process. Only
            applicable when running on a multi-node system.
        wall_time: Maximum time to allow job to run for. Only applicable when submitting
            to a job scheduler.
        create_missing_parameters: Force missing parameters in overrides to be added
        **parameter_overrides: Additional keyword arguments will be passed to
            ``FabNESO.ensemble_tools.edit_parameters`` to create a temporary conditions
            file with these parameter vaues overriden.
    """
    _check_fab_module_imported()
    processes, nodes, cpus_per_process, wall_time = _check_and_process_resource_args(
        processes, nodes, cpus_per_process, wall_time
    )
    # Use a temporary directory context so that we can handle parameter inputs
    # from the command line
    original_config_path = Path(fab.find_config_file_path(config))
    temporary_context: TemporaryDirectory | nullcontext = (
        TemporaryDirectory(prefix=f"{config}_", dir=original_config_path.parent)
        if parameter_overrides != {}
        else nullcontext()
    )
    with temporary_context as temporary_config_directory:
        # If there have been additional parameters provided, create a copy of the
        # conditions file and edit the input parameters
        if parameter_overrides != {}:
            temporary_config_path = Path(temporary_config_directory)
            shutil.copytree(
                original_config_path, temporary_config_path, dirs_exist_ok=True
            )
            config = temporary_config_path.name  # switch our config to the new tmp ones
            edit_parameters(
                temporary_config_path / conditions_file_name,
                parameter_overrides,
                create_missing=create_missing_parameters,
            )

        fab.with_config(config)
        fab.execute(fab.put_configs, config)
        fab.job(
            _create_job_args_dict(
                solver,
                conditions_file_name,
                mesh_file_name,
                processes,
                nodes,
                cpus_per_process,
                wall_time,
            )
        )


def _parse_parameter_scan_string(
    parameter_scan_string: str,
    delimiter: str,
) -> tuple[float, float, int]:
    start, end, n_steps = parameter_scan_string.split(delimiter)
    return float(start), float(end), int(n_steps)


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_ensemble(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    processes: int = 4,
    nodes: int = 1,
    cpus_per_process: int = 1,
    wall_time: str = "00:15:00",
    **parameter_scans: str,
) -> None:
    """
    Run ensemble of NESO solver instances.

    Args:
        config: Directory with ensemble configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        processes: Number of processes to run in each job in the ensemble.
        nodes: Number of nodes to run each job in ensemble on. Only applicable when
            running on a multi-node system.
        cpus_per_process: Number of processing units to use per process for each job in
            ensemble. Only applicable when running on a multi-node system.
        wall_time: Maximum time to allow each job in ensemble to run for. Only
            applicable when submitting to a job scheduler.
        **parameter_scans: The set of parameters to sweep over. A colon separated list
            of lower bound, upper bound, and steps.
    """
    _check_fab_module_imported()
    processes, nodes, cpus_per_process, wall_time = _check_and_process_resource_args(
        processes, nodes, cpus_per_process, wall_time
    )
    path_to_config = Path(fab.find_config_file_path(config))
    temporary_context: TemporaryDirectory | nullcontext = (
        TemporaryDirectory(prefix=f"{config}_", dir=path_to_config.parent)
        if parameter_scans != {}
        else nullcontext()
    )
    with temporary_context as temporary_config_directory:
        if parameter_scans != {}:
            temporary_config_path = Path(temporary_config_directory)
            # Because FabSIM is a bit weird with commas, build the dict here
            parameter_scan_dict = {
                parameter: _parse_parameter_scan_string(values, ":")
                for parameter, values in parameter_scans.items()
            }
            create_dict_sweep(
                sweep_path=temporary_config_path,
                destructive=False,
                copy_dir=path_to_config,
                edit_file=conditions_file_name,
                parameter_dict=parameter_scan_dict,
            )

            # switch our config to the new tmp ones
            config = temporary_config_path.name
            path_to_config = temporary_config_path

        sweep_dir = str(path_to_config / "SWEEP")
        fab.update_environment(
            _create_job_args_dict(
                solver,
                conditions_file_name,
                mesh_file_name,
                processes,
                nodes,
                cpus_per_process,
                wall_time,
            )
        )
        fab.with_config(config)
        fab.run_ensemble(config, sweep_dir)


def _parse_vbmc_bounds_string(
    vbmc_bounds_string: str,
    delimiter: str,
) -> tuple[float, float]:
    lower_bound, upper_bound = vbmc_bounds_string.split(delimiter)
    return float(lower_bound), float(upper_bound)


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_vbmc(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    processes: int = 4,
    nodes: int = 1,
    cpus_per_process: int = 1,
    wall_time: str = "00:15:00",
    output_directory_name: str = "",
    reference_field_file: str = "",
    **vbmc_parameters: str,
) -> None:
    """
    Run an instance of PyVBMC on a NESO solver.

    Args:
        config: Directory with ensemble configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        processes: Number of processes to run in each job in the ensemble.
        nodes: Number of nodes to run on. Only applicable when running on a multi-node
            system.
        cpus_per_process: Number of processing units to use per process. Only
            applicable when running on a multi-node system.
        wall_time: Maximum time to allow job to run for. Only applicable when submitting
            to a job scheduler.
        reference_field_file: Name of a numpy file that holds a reference field
            measurement for the calibration run
        **vbmc_parameters: The parameters to be scanned in the VBMC instance. The value
            is a colon separated list of the lower and upper bounds of the parameter

    """
    # Create the output directory
    output_directory = Path(output_directory_name)
    if not output_directory.is_dir():
        output_directory.mkdir(parents=True)

    if not Path(reference_field_file).is_file():
        msg = (
            f"reference_field_file {reference_field_file} not found. "
            "One must be specificed for the calibration to run. If in doubt, please "
            "run the neso_write_field task to write a measured field to file first."
        )
        raise ValueError(msg)

    # Contains a number of items that enable the vbmc running of NESO

    # These ensure that the field information is written out, which is not done
    # be default in the conditions files.
    # This would need to be changed per solver
    para_overrides = {
        "particle_num_write_field_steps": 100,
        "line_field_deriv_evaluations_step": 20,
        "line_field_deriv_evaluations_numx": 100,
        "line_field_deriv_evaluations_numy": 1,
    }

    # Put all of the NESO arguments in one dict
    neso_args = _create_job_args_dict(
        solver,
        conditions_file_name,
        mesh_file_name,
        processes,
        nodes,
        cpus_per_process,
        wall_time,
    )

    # Make this read in from a config too?
    observation_noise_std = 0.1

    # Make the config_dict for the calibration run
    config_dict = {
        "config": config,
        "para_overrides": para_overrides,
        "neso_args": neso_args,
        "observation_noise_std": observation_noise_std,
    }

    # Hard coded for the two_stream config. Ideally this would be factored out
    parameters_to_scan = {
        vbmc_parameter: _parse_vbmc_bounds_string(parameter_bounds, ":")
        for vbmc_parameter, parameter_bounds in vbmc_parameters.items()
    }

    config_dict["parameters_to_scan"] = parameters_to_scan

    # Set up the bounds information required by PyVBMC
    bounds = list(zip(*parameters_to_scan.values(), strict=True))
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])

    plausible_lower_bounds = lower_bounds + (upper_bounds - lower_bounds) / 4
    plausible_upper_bounds = upper_bounds - (upper_bounds - lower_bounds) / 4

    # Choose a random starting position
    rng = np.random.default_rng()
    theta_0 = rng.uniform(plausible_lower_bounds, plausible_upper_bounds)

    # Read in a reference field from an external file for the calibration
    config_dict["initial_run"] = np.loadtxt(reference_field_file)

    # Get a timestamp to label the vbmc logfile
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile_name = f"vbmc_log_{timestamp}.log"
    logfile_path = Path(output_directory) / logfile_name

    # Additional options that we will pass to VBMC
    options = {
        # 50 * D + 2 is the default, but leave this here to allow
        # smaller runs for debugging
        "max_fun_evals": 50 * (len(theta_0) + 2),
        # Can't see a way of doing this and saving the plots
        # without it creating a new xwindow?
        "plot": False,
        # Run with a log file
        "log_file_name": logfile_path,
    }

    # Make an instance of VBMC
    vbmc = pyvbmc.VBMC(
        lambda theta: log_density(theta, config_dict),
        theta_0,
        lower_bounds,
        upper_bounds,
        plausible_lower_bounds,
        plausible_upper_bounds,
        options=options,
    )

    # Run the vbmc instance
    vp, results = vbmc.optimize()

    # Print the final output of vbmc to the logfile
    with logfile_path.open("a+") as logfile:
        logfile.write(pyvbmc.formatting.format_dict(results))

    # Save the final output posterior
    vbmc.vp.save(output_directory / f"final_posterior_{timestamp}.pkl")


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_write_field(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    processes: int = 4,
    nodes: int = 1,
    cpus_per_process: int = 1,
    wall_time: str = "00:15:00",
    out_file_name: str = "field_write_out.txt",
    **parameter_overrides: str,
) -> None:
    """
    Run a single NESO solver instance and save the observed field to a file.

    Args:
        config: Directory with single run configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        processes: Number of processes to run.
        nodes: Number of nodes to run on. Only applicable when running on a multi-node
            system.
        cpus_per_process: Number of processing units to use per process. Only
            applicable when running on a multi-node system.
        wall_time: Maximum time to allow job to run for. Only applicable when submitting
            to a job scheduler.
        out_file_name: Name of the file to be created containing the measured field
        **parameter_overrides: Additional keyword arguments will be passed to
            ``FabNESO.ensemble_tools.edit_parameters`` to create a temporary conditions
            file with these parameter vaues overriden.
    """
    # Assemble the NESO arguments
    neso_args = _create_job_args_dict(
        solver,
        conditions_file_name,
        mesh_file_name,
        processes,
        nodes,
        cpus_per_process,
        wall_time,
    )

    para_overrides = dict(
        {
            # The necessary overrides for writing out the field.
            # Factor this out of this and VBMC task
            "particle_num_write_field_steps": 100,
            "line_field_deriv_evaluations_step": 20,
            "line_field_deriv_evaluations_numx": 100,
            "line_field_deriv_evaluations_numy": 1,
        },
        **parameter_overrides,
    )

    config_dict = {"config": config, "neso_args": neso_args}

    # Write out the returned field to file
    np.savetxt(
        out_file_name,
        run_instance_return_field(config_dict, para_overrides)["field_value"],
    )


def run_instance_return_field(
    config_dict: dict[str, Any], para_overrides: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Run a single instance of the NESO solver and return the observed_field."""
    # If we're running remotely, tell the submission to wait until done
    if "archer2" in fab.env.remote:
        fab.update_environment(
            {
                "job_dispatch": "cd /work/$project/$project/$username ; sbatch --wait",
            }
        )
    neso(
        config=config_dict["config"],
        solver=config_dict["neso_args"]["neso_solver"],
        conditions_file_name=config_dict["neso_args"]["neso_conditions_file"],
        mesh_file_name=config_dict["neso_args"]["neso_mesh_file"],
        processes=config_dict["neso_args"]["cores"],
        nodes=config_dict["neso_args"]["nodes"],
        cpus_per_process=config_dict["neso_args"]["cpuspertask"],
        wall_time=config_dict["neso_args"]["job_wall_time"],
        create_missing_parameters=True,
        **para_overrides,
    )
    fab.fetch_results()
    local_results_dir = Path(fab.env.job_results_local) / template(
        fab.env.job_name_template
    )
    final_line_field_step = (
        int(
            list_parameter_values(
                Path(fab.find_config_file_path(config_dict["config"]))
                / str(config_dict["neso_args"]["neso_conditions_file"]),
                "particle_num_time_steps",
            )[0]
        )
        - para_overrides["line_field_deriv_evaluations_step"]
    )
    return read_hdf5_datasets(
        local_results_dir / "Electrostatic2D3V_line_field_deriv_evaluations.h5part",
        {
            "x": f"Step#{final_line_field_step}/x",
            "field_value": f"Step#{final_line_field_step}/FIELD_EVALUATION_0",
        },
    )


def log_density(
    theta: list[float],
    config_dict: dict[str, Any],
) -> list:
    """Run an instance of the neso task and return the log_joint_density."""
    parameters = dict(
        zip(config_dict["parameters_to_scan"].keys(), theta, strict=True),
        **config_dict["para_overrides"],
    )

    # Run an instance of NESO and calculate the measured field strength.
    observed_results = run_instance_return_field(config_dict, parameters)

    # Calculate the joint log likelihood using the reference field in the config_dict
    return -(
        (config_dict["initial_run"] - observed_results["field_value"]) ** 2
        / (2 * config_dict["observation_noise_std"] ** 2)
    ).sum()
