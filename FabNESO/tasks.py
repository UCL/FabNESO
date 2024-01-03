"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

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


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    **parameter_overrides: str,
) -> None:
    """
    Run a single NESO solver instance.

    Args:
        config: Directory with single run configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
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
                temporary_config_path / conditions_file_name, parameter_overrides
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
    **parameter_scans: str,
) -> None:
    """
    Run ensemble of NESO solver instances.

    Args:
        config: Directory with ensemble configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        **parameter_scans: The set of parameters to sweep over. A colon separated list
        of lower bound, upper bound, and steps.
    """
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
            {
                "script": "neso",
                "neso_solver": solver,
                "neso_conditions_file": conditions_file_name,
                "neso_mesh_file": mesh_file_name,
            }
        )
        fab.with_config(config)
        fab.run_ensemble(config, sweep_dir)
