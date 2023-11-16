# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is
# distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabNeso
#
# authors: Duncan Leggat

try:
    import fabsim.base.fab as fab
except ImportError:
    import base.fab as fab

from pathlib import Path

fab.add_local_paths("FabNeso")


@fab.task
def neso(
    config,
    solver="views/gcc-hipsycl/bin/Electrostatic2D3V",
    conditions_file_name="two_stream_conditions.xml",
    mesh_file_name="two_stream_mesh.xml",
    **args
):
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

    fab.job(dict(script="neso", wall_time="0:15:0", memory="2G"), args)


@fab.task
def neso_ensemble(
    config,
    solver="views/gcc-hipsycl/bin/Electrostatic2D3V",
    conditions_file_name="two_stream_conditions.xml",
    mesh_file_name="two_stream_mesh.xml",
    **args
):
    path_to_config = fab.find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    fab.env.script = "neso"

    fab.env.neso_solver = solver
    fab.env.neso_conditions_file = conditions_file_name
    fab.env.neso_mesh_file = mesh_file_name

    fab.with_config(config)
    fab.run_ensemble(config, sweep_dir, **args)
