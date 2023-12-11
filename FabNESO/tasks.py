"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

import shutil
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pyvbmc

try:
    from fabsim.base import fab
except ImportError:
    from base import fab

try:
    from fabsim.deploy.templates import template
except ImportError:
    from deploy.templates import template

from .ensemble_tools import edit_parameters
from .read_outputs import read_hdf5_datasets

fab.add_local_paths("FabNESO")


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
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
        create_missing_parameters: Force missing parameters in overrides to be added
        **parameter_overrides: Additional keyword arguments will be passed to
            ``FabNESO.ensemble_tools.edit_parameters`` to create a temporary conditions
            file with these parameter vaues overriden.
    """
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
            {
                "script": "neso",
                "neso_solver": solver,
                "neso_conditions_file": conditions_file_name,
                "neso_mesh_file": mesh_file_name,
            }
        )


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_ensemble(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
) -> None:
    """
    Run ensemble of NESO solver instances.

    Args:
        config: Directory with ensemble configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
    """
    path_to_config = fab.find_config_file_path(config)
    sweep_dir = str(Path(path_to_config) / "SWEEP")
    fab.update_environment(
        {
            "script": "neso",
            "neso_solver": solver,
            "neso_conditions_file": conditions_file_name,
            "neso_mesh_file": mesh_file_name,
        }
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

    config_dict["parameters_to_scan"]

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
    config_dict: dict[str, object],
) -> dict[str, np.ndarray]:
    """Run a single instance of the NESO solver and return the observed_field."""
    neso(
        config=config_dict["config"],
        solver=config_dict["neso_solver"],
        conditions_file_name=config_dict["neso_conditions_file"],
        mesh_file_name=config_dict["neso_mesh_file"],
        create_missing_parameters=True,
        **config_dict["para_overrides"],
    )
    fab.fetch_results()
    local_results_dir = Path(fab.env.job_results_local) / template(
        fab.env.job_name_template
    )
    final_line_field_step = 180
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
