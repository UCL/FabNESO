"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

import json
import pickle
import re
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypeAlias

import chaospy as cp
import numpy as np
import pandas as pd
import pyvbmc
from easyvvuq.analysis import PCEAnalysis
from easyvvuq.sampling import PCESampler

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


from .ensemble_tools import (
    SamplingRule,
    create_grid_ensemble,
    create_qmc_ensemble,
    edit_parameters,
    list_parameter_values,
)
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

    Keyword Args:
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
        create_missing_parameters: Force parameters in ``parameter_overrides`` missing
            from conditions file to be added.
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


def _parse_parameter_range_string(
    parameter_range_string: str,
    delimiter: str,
) -> tuple[float, float, int]:
    lower, upper, n_sample = parameter_range_string.split(delimiter)
    return float(lower), float(upper), int(n_sample)


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_grid_ensemble(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    processes: int = 4,
    nodes: int = 1,
    cpus_per_process: int = 1,
    wall_time: str = "00:15:00",
    **parameter_ranges: str,
) -> None:
    """
    Run ensemble of NESO solver instances on a evenly spaced parameter grid.

    Args:
        config: Directory with ensemble configuration information.

    Keyword Args:
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
        **parameter_ranges: The parameter ranges to construct grid over. For each
            parameter name, a string of the format `lower:upper:n_sample` should be
            specified, resulting `n_sample` evenly spaced values over the interval
            `[lower, upper]`. The overall grid is constructed as the tensor product of
            the samples for each parameter.

    """
    _check_fab_module_imported()
    processes, nodes, cpus_per_process, wall_time = _check_and_process_resource_args(
        processes, nodes, cpus_per_process, wall_time
    )
    source_config_path = Path(fab.find_config_file_path(config))
    temporary_context = TemporaryDirectory(
        prefix=f"{config}_", dir=source_config_path.parent
    )
    with temporary_context as temporary_config_directory:
        temporary_config_path = Path(temporary_config_directory)
        output_path = temporary_config_path / "SWEEP"
        parsed_parameter_ranges = {
            parameter: _parse_parameter_range_string(values, ":")
            for parameter, values in parameter_ranges.items()
        }
        create_grid_ensemble(
            output_path=output_path,
            source_path=source_config_path,
            conditions_file=conditions_file_name,
            parameter_ranges=parsed_parameter_ranges,
        )
        config = temporary_config_path.name
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
        fab.run_ensemble(config, str(output_path))


def _parse_parameter_interval_string(
    parameter_interval_string: str,
    delimiter: str,
) -> tuple[float, float]:
    lower, upper = parameter_interval_string.split(delimiter)
    return float(lower), float(upper)


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_qmc_ensemble(  # noqa: PLR0913
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    processes: int = 4,
    nodes: int = 1,
    cpus_per_process: int = 1,
    wall_time: str = "00:15:00",
    n_sample: int = 100,
    seed: int = 1234,
    rule: SamplingRule = "latin_hypercube",
    **parameter_intervals: str,
) -> None:
    """
    Run ensemble of NESO solver instances on quasi-Monte Carlo parameter samples.

    Args:
        config: Directory with conditions and mesh files to create ensemble from.

    Keyword Args:
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
        n_sample: Number of quasi Monte Carlo samples in ensemble.
        seed: Seed for pseudo-random number generator.
        rule: String specifying sampling scheme to use.
        **parameter_intervals: The parameter intervals over which to generate samples
            from. For each parameter name, a string of the format `lower:upper` should
            be specified, with the overall joint distribution on the parameter space the
            product of the uniform distributions on these intervals.

    """
    _check_fab_module_imported()
    n_sample = _try_convert_to_int_and_check_positive(n_sample, "n_sample")
    seed = _try_convert_to_int_and_check_positive(seed, "seed")
    processes, nodes, cpus_per_process, wall_time = _check_and_process_resource_args(
        processes, nodes, cpus_per_process, wall_time
    )
    source_config_path = Path(fab.find_config_file_path(config))
    temporary_context = TemporaryDirectory(
        prefix=f"{config}_", dir=source_config_path.parent
    )
    with temporary_context as temporary_config_directory:
        temporary_config_path = Path(temporary_config_directory)
        parsed_parameter_intervals = {
            parameter: _parse_parameter_interval_string(interval_string, ":")
            for parameter, interval_string in parameter_intervals.items()
        }
        output_path = temporary_config_path / "SWEEP"
        create_qmc_ensemble(
            output_path=output_path,
            source_path=source_config_path,
            conditions_file=conditions_file_name,
            parameter_intervals=parsed_parameter_intervals,
            rule=rule,
            n_sample=n_sample,
            seed=seed,
        )
        config = temporary_config_path.name
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
        fab.run_ensemble(config, str(output_path))


def _parse_vbmc_bounds_string(
    vbmc_bounds_string: str,
    delimiter: str,
) -> tuple[float, float, float, float]:
    (
        lower_bound,
        upper_bound,
        plausible_lower_bound,
        plausible_upper_bound,
    ) = vbmc_bounds_string.split(delimiter)
    return (
        float(lower_bound),
        float(upper_bound),
        float(plausible_lower_bound),
        float(plausible_upper_bound),
    )


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_vbmc(  # noqa: PLR0913
    config: str,
    reference_field_file: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    observation_noise_std: float = 0.1,
    processes: int = 4,
    nodes: int = 1,
    cpus_per_process: int = 1,
    wall_time: str = "00:15:00",
    output_directory_name: str = "",
    **vbmc_parameters: str,
) -> None:
    """
    Run variational Bayesian Monte Carlo (VBMC) to calibrate NESO solver parameters.

    The VBMC algorithm (Acerbi, 2018) is an approximate Bayesian inference method for
    efficient parameter calibration in expensive to simulate models. Here we use the
    Python implementation of the algorithm in the package PyVBMC (Huggins et al., 2023).

    Args:
        config: Directory with ensemble configuration information.
        reference_field_file: Name of a NumPy .txt file that holds a reference
            field measurement for the calibration run.

    Keyword Args:
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        observation_noise_std: Standard deviation of the observed noise, used for log
            likelihood calculation.
        processes: Number of processes to run in each job in the ensemble.
        nodes: Number of nodes to run on. Only applicable when running on a multi-node
            system.
        cpus_per_process: Number of processing units to use per process. Only
            applicable when running on a multi-node system.
        wall_time: Maximum time to allow job to run for. Only applicable when submitting
            to a job scheduler.
        **vbmc_parameters: The parameters to be scanned in the VBMC instance. The value
            is a colon separated list: lower bound: upper bound: plausible lower bound
            : plausible upper bound

    References:
        1. Acerbi, L. (2018). Variational Bayesian Monte Carlo.
           Advances in Neural Information Processing Systems, 31.
        2. Huggins et al., (2023). PyVBMC: Efficient Bayesian inference in Python.
           Journal of Open Source Software, 8(86), 5428,
           https://doi.org/10.21105/joss.05428

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
    plausible_lower_bounds = np.array(bounds[2])
    plausible_upper_bounds = np.array(bounds[3])

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
        lambda theta: _log_density(theta, config_dict),
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

    Keyword Args:
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
        _run_instance_return_field(config_dict, para_overrides)["field_value"],
    )


def _run_instance_return_field(
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


def _log_density(
    theta: list[float],
    config_dict: dict[str, Any],
) -> list:
    """Run an instance of the neso task and return the log_joint_density."""
    parameters = dict(
        zip(config_dict["parameters_to_scan"].keys(), theta, strict=True),
        **config_dict["para_overrides"],
    )

    # Run an instance of NESO and calculate the measured field strength.
    observed_results = _run_instance_return_field(config_dict, parameters)

    # Calculate the joint log likelihood using the reference field in the config_dict
    return -(
        (config_dict["initial_run"] - observed_results["field_value"]) ** 2
        / (2 * config_dict["observation_noise_std"] ** 2)
    ).sum()


def _parse_pce_bounds_string(
    parameter_scan_string: str,
    delimiter: str,
) -> tuple[float, float]:
    lower, upper = parameter_scan_string.split(delimiter)
    return float(lower), float(upper)


def _parse_float_or_int_string_literal(literal: str) -> float | int:
    try:
        return int(literal)
    except ValueError:
        return float(literal)


PCEVariant: TypeAlias = Literal[
    "pseudo-spectral", "pseudo-spectral-sparse", "point-collocation"
]


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_pce_ensemble(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    polynomial_order: int = 4,
    variant: PCEVariant = "pseudo-spectral",
    processes: str | int = 4,
    nodes: str | int = 1,
    cpus_per_process: str | int = 1,
    wall_time: str = "00:15:00",
    **parameter_bounds_or_overrides: str,
) -> None:
    """
    Run ensemble of NESO simulations to perform a polynomial chaos expansion of outputs.

    Generates a set of parameters values (and associated weights) to evaluate model at
    using a quadrature rule, and evaluates model outputs at each of these parameter
    values. The model outputs can then be approximated by an expansion in a set of
    orthogonal (with respect to the assumed distribution over the parameter space)
    polynomials, with the coefficients of the expansion estimated from the sampled
    model outputs. This task just computes the model outputs for the sampled parameter
    values, with the separate `neso_pce_analysis` task using the fetched model outputs
    from this task to estimate the coefficients and so form the polynomial expansion
    approximation to the model.

    Args:
        config: Directory with ensemble configuration information.

    Keyword Args:
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        polynomial_order: Polynomial order to use in polynomial chaos expansion.
        variant: Polynomial chaos expansion variant to use - one of `point-collocation`
            (point-collocation method), `pseudo-spectral` (pseudo-spectral projection
            method) or `pseudo-spectral-sparse` (pseudo-spectral projection method with
            Smolyak sparse grid).
        processes: Number of processes to run in each job in the ensemble.
        nodes: Number of nodes to run on. Only applicable when running on a multi-node
            system.
        cpus_per_process: Number of processing units to use per process. Only
            applicable when running on a multi-node system.
        wall_time: Maximum time to allow job to run for. Only applicable when submitting
            to a job scheduler.
        **parameter_bounds_or_overrides: Bounds of parameters to vary in polynomial
            chaos expansion or fixed overrides for parameters from values in conditions
            file in configuration directory. Each value is either a colon-separated
            string `lower_bound:upper_bound` specifying lower and upper bounds for
            independent uniform distributions over parameter values, or a string
            specifying a single int or float, in which case the corresponding parameter
            is considered fixed but the value given is used to override its default
            value in the conditions file.

    """
    _check_fab_module_imported()
    parameter_distributions = {
        parameter_name: cp.Uniform(*_parse_pce_bounds_string(string_value, ":"))
        for parameter_name, string_value in parameter_bounds_or_overrides.items()
        if ":" in string_value
    }
    parameter_overrides = {
        parameter_name: _parse_float_or_int_string_literal(string_value)
        for parameter_name, string_value in parameter_bounds_or_overrides.items()
        if ":" not in string_value
    }
    # Map variant specifier to EasyVVUQ PCESampler keyword arguments - regression is
    # used to switch between point-collocation (True) and pseudo-spectral (False)
    # methods, while sparse (True) enables Smolyak sparse grid with pseudo-spectral
    # method.
    match variant:
        case "point-collocation":
            regression = True
            sparse = False
        case "pseudo-spectral":
            regression = False
            sparse = False
        case "pseudo-spectral-sparse":
            regression = False
            sparse = True
    pce_sampler = PCESampler(
        parameter_distributions,
        polynomial_order=int(polynomial_order),
        regression=regression,
        sparse=sparse,
    )
    parameter_samples = list(pce_sampler)
    processes, nodes, cpus_per_process, wall_time = _check_and_process_resource_args(
        processes, nodes, cpus_per_process, wall_time
    )
    path_to_config = Path(fab.find_config_file_path(config))
    with TemporaryDirectory(
        prefix=f"{config}_", dir=path_to_config.parent
    ) as temporary_config_directory:
        temporary_config_path = Path(temporary_config_directory)
        for sample_index, parameter_dict in enumerate(parameter_samples):
            directory_path = temporary_config_path / "SWEEP" / f"sample_{sample_index}"
            shutil.copytree(path_to_config, directory_path)
            edit_parameters(
                directory_path / conditions_file_name,
                parameter_dict | parameter_overrides,
            )
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
        job_name = template(fab.env.job_name_template)
        local_results_directory = Path(fab.env.local_results) / job_name
        local_results_directory.mkdir(parents=True)
        with (local_results_directory / "pce_sampler.pickle").open("wb") as f:
            pickle.dump(pce_sampler, f)
        with (local_results_directory / "parameter_samples.json").open("w") as f:
            json.dump(parameter_samples, f)
        print(  # noqa: T201
            f"Sampler pickle and parameters JSON saved to {local_results_directory}.\n"
            "Run fetch_results task once runs are completed to also save run outputs "
            "to same directory and then run neso_pce_analysis task to compute "
            "PCE expansion from run outputs."
        )


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_pce_analysis(
    config: Path | str,
    results_dir: Path | str,
    extract_outputs_script: str = "extract_outputs.py",
) -> None:
    """
    Analyse outputs from a previous polynomial chaos expansion (PCE) ensemble run.

    Uses run outputs for sampled parameter values to compute a PCE approximation to
    model, and uses this to compute various statistics of output and allowing
    construction of a surrogate model. The analysis results are saved to a pickle file
    `pce_analysis_results.pickle` in the results directory.

    Args:
        config: Name of configuration directory with script to use to extract relevant
            solver outputs from results files and output to a JSON file.
        results_dir: Directory containing PCE ensemble outputs from a run of `neso_pce`
            task. The analysis results pickle file will be written to this directory.

    Keyword Args:
        extract_outputs_script: Name of script for extracting outputs from results
           files in configuration directory.

    """
    _check_fab_module_imported()
    path_to_config = Path(fab.find_config_file_path(config))
    extract_outputs_script_path = path_to_config / extract_outputs_script
    results_dir = Path(results_dir)
    with (results_dir / "pce_sampler.pickle").open("rb") as f:
        pce_sampler = pickle.load(f)  # noqa: S301
    with (results_dir / "parameter_samples.json").open("r") as f:
        parameter_samples = json.load(f)
    results_data = []
    for sample_index, parameter_dict in enumerate(parameter_samples):
        sample_results_dir = results_dir / "RUNS" / f"sample_{sample_index}"
        outputs_file = sample_results_dir / "outputs.json"
        subprocess.call(
            [  #  noqa: S603, S607
                "python",
                str(extract_outputs_script_path),
                sample_results_dir,
                outputs_file,
            ]
        )
        with outputs_file.open("r") as f:
            outputs = json.load(f)
        results_data.append(
            {(key, i): v for key, value in outputs.items() for i, v in enumerate(value)}
            | {(key, 0): value for key, value in parameter_dict.items()}
        )
    results_dataframe = pd.DataFrame(
        results_data, columns=pd.MultiIndex.from_tuples(results_data[0].keys())
    )
    pce_analysis = PCEAnalysis(sampler=pce_sampler, qoi_cols=list(outputs.keys()))
    analysis_results = pce_analysis.analyse(results_dataframe)
    with (results_dir / "pce_analysis_results.pickle").open("wb") as f:
        pickle.dump(analysis_results, f)
