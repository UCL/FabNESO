"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

import re
import shutil
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
    template = SimpleNamespace(
        task=lambda f: f, load_plugin_env_vars=lambda _: lambda f: f
    )
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


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_vbmc(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
) -> None:
    """Run an instance of PyVBMC on a NESO solver."""
    # This config dict should probably at some point be factored out
    config_dict = {
        "config": config,
        "neso_solver": solver,
        "neso_conditions_file": conditions_file_name,
        "neso_mesh_file": mesh_file_name,
        "para_overrides": {
            # These ensure that the field information is written out
            "particle_num_write_field_steps": 100,
            "line_field_deriv_evaluations_step": 20,
            "line_field_deriv_evaluations_numx": 100,
            "line_field_deriv_evaluations_numy": 1,
        },
        "noise_factor": 0.1,
    }

    # I want to define this inside the config_dict, but mypy simply will not allow it
    parameters_to_scan = {
        "particle_initial_velocity": (0.0, 2.0),
        "particle_charge_density": (20.0, 200.0),
        "particle_number_density": (20.0, 200.0),
    }

    config_dict["parameters_to_scan"] = parameters_to_scan

    bounds = list(zip(*parameters_to_scan.values(), strict=True))
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])

    plausible_lower_bounds = lower_bounds + (upper_bounds - lower_bounds) / 4
    plausible_upper_bounds = upper_bounds - (upper_bounds - lower_bounds) / 4
    # Choose a random starting position
    rng = np.random.default_rng()
    theta_0 = rng.uniform(plausible_lower_bounds, plausible_upper_bounds)

    # Run a nominal version of the model with central values to find
    # the true field measurements. Important for the log calculation
    config_dict["initial_run"] = run_instance_return_field(config_dict)

    # Make an instance of VBMC
    vbmc = pyvbmc.VBMC(
        lambda theta: log_density(theta, config_dict),
        theta_0,
        lower_bounds,
        upper_bounds,
        plausible_lower_bounds,
        plausible_upper_bounds,
        options={"plot": True},
    )

    # Run the vbmc instance
    vp, results = vbmc.optimize()


def run_instance_return_field(
    config_dict: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Run a single instance of the NESO solver and return the observed_field."""
    neso(
        config=config_dict["config"],
        solver=config_dict["neso_solver"],
        conditions_file_name=config_dict["neso_conditions_file"],
        mesh_file_name=config_dict["neso_mesh_file"],
        processes=1,
        create_missing_parameters=True,
        **config_dict["para_overrides"],
    )
    fab.fetch_results()
    local_results_dir = Path(fab.env.job_results_local) / template(
        fab.env.job_name_template
    )
    final_line_field_step = (
        int(
            list_parameter_values(
                Path(fab.find_config_file_path(config_dict["config"]))
                / str(config_dict["neso_conditions_file"]),
                "particle_num_time_steps",
            )[0]
        )
        - config_dict["para_overrides"]["line_field_deriv_evaluations_step"]
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

    config_dict["para_overrides"] = parameters
    observed_results = run_instance_return_field(config_dict)
    return -(
        (config_dict["initial_run"]["field_value"] - observed_results["field_value"])
        ** 2
        / (2 * config_dict["noise_factor"] ** 2)
    ).sum()
