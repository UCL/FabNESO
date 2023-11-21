"""
Task definitions for FabNESO plug-in to FabSIM software toolkit.

Defines tasks for running simulations using Neptune Exploratory Software (NESO).
"""

from pathlib import Path

try:
    import fabsim.base.fab as fab
except ImportError:
    import base.fab as fab


fab.add_local_paths("FabNeso")


@fab.task
@fab.load_plugin_env_vars("FabNeso")
def neso(
    config,
    solver="views/gcc-hipsycl/bin/Electrostatic2D3V",
    conditions_file_name="two_stream_conditions.xml",
    mesh_file_name="two_stream_mesh.xml",
    wall_time_in="0:15:0",
    memory_in="2G",
    **args
):
    """
    Run one NESO instance1

    parameters:
      - config: Directory with configuration information
      - solver: Which NESO solver to use
      - conditions_file_name: Path to conditions file
      - mesh_file_name: Path to mesh file for NESO
    """
    fab.update_environment(args)
    fab.with_config(config)
    fab.execute(fab.put_configs, config)

    fab.env.neso_solver = solver

    # This we presumably change somehow so that it gets changed throughout
    # the SWEEP dir?
    fab.env.neso_conditions_file = (
        Path(fab.find_config_file_path(config)) / conditions_file_name
    )
    # All of these should be in a config file somewhere
    fab.env.neso_mesh_file = Path(fab.find_config_file_path(config)) / mesh_file_name

    fab.job(dict(script="neso", wall_time=wall_time_in, memory=memory_in), args)


@fab.task
@fab.load_plugin_env_vars("FabNeso")
def neso_ensemble(
    config,
    solver="views/gcc-hipsycl/bin/Electrostatic2D3V",
    conditions_file_name="two_stream_conditions.xml",
    mesh_file_name="two_stream_mesh.xml",
    **args
):
    """
    Run NESO ensemble

    parameters:
      - config: Directory containing SWEEP configurations
      - solver: Which NESO solver to use
      - conditions_file_name: Path to conditions file
      - mesh_file_name: Path to mesh file for NESO
    """
    path_to_config = fab.find_config_file_path(config)
    sweep_dir = str((Path(path_to_config) / "SWEEP"))
    fab.env.script = "neso"

    fab.env.neso_solver = solver
    fab.env.neso_conditions_file = conditions_file_name
    fab.env.neso_mesh_file = mesh_file_name

    fab.with_config(config)
    fab.run_ensemble(config, sweep_dir, **args)
