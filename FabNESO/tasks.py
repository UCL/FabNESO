"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from fabsim.base import fab
except ImportError:
    from base import fab


from .ensemble_tools import create_dict_sweep, edit_parameters

fab.add_local_paths("FabNESO")


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
    **parameter_overrides: str,
) -> None:
    """
    Run a single NESO solver instance.

    Args:
        config: Directory with single run configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        processes: Number of processes to run on.
        nodes: Number of nodes to run on. Only applicable when running on a multi-node
            system.
        cpus_per_process: Number of processing units to use per process. Only
            applicable when running on a multi-node system.
        wall_time: Maximum time to allow job to run for. Only applicable when submitting
            to a job scheduler.
        **parameter_overrides: Additional keyword arguments will be passed to
            ``FabNESO.ensemble_tools.edit_parameters`` to create a temporary conditions
            file with these parameter vaues overriden.
    """
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
                temporary_config_path / conditions_file_name, parameter_overrides
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
