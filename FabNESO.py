"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

from pathlib import Path
from plugins.FabNESO.utils.ensemble_tools import edit_parameters
import shutil

try:
    import fabsim.base.fab as fab
except ImportError:
    import base.fab as fab  # type: ignore


fab.add_local_paths("FabNESO")


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
    **kwargs
):
    """
    Run a single NESO solver instance.

    Args:
        config: Directory with single run configuration information.
        solver: Which NESO solver to use.
        conditions_file_name: Name of conditions XML file in configuration directory.
        mesh_file_name: Name of mesh XML in configuration directory.
        **kwargs: Additional kwargs will be passed to ensemble_tools.edit_parameters to create a temporary conditions file with these overriden parameters
    """
    # If there have been additional parameters provided, create a copy of the conditions file and edit the input parameters
    if not kwargs == {}:
        shutil.copytree(fab.find_config_file_path(config),Path(fab.find_config_file_path(config)).parent / "tmp_configs")
        config = "tmp_configs" #switch our config to the new tmp ones
        edit_parameters(Path(fab.find_config_file_path(config)) / conditions_file_name, kwargs)
            
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

    # Remove the temporary configurations, if any
    if not kwargs == {}:
        shutil.rmtree(fab.find_config_file_path(config))


@fab.task
@fab.load_plugin_env_vars("FabNESO")
def neso_ensemble(
    config: str,
    solver: str = "Electrostatic2D3V",
    conditions_file_name: str = "conditions.xml",
    mesh_file_name: str = "mesh.xml",
):
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
