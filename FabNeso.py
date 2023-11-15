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
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

add_local_paths("FabNeso")

@task
def neso(config,
         solver="views/gcc-hipsycl/bin/Electrostatic2D3V",
         conditionsFileName="two_stream_conditions.xml",
         meshFileName="two_stream_mesh.xml",
         **args):

    update_environment(args)
    with_config(config)
    execute(put_configs, config)

    env.neso_solver = solver

    #This we presumably change somehow so that it gets changed throughout
    #the SWEEP dir?
    env.neso_conditions_file = (find_config_file_path(config)+
                                "/" + conditionsFileName)
    #All of these should be in a config file somewhere
    env.neso_mesh_file = find_config_file_path(config) + "/" + meshFileName

    
    job(dict(script='neso', wall_time='0:15:0', memory='2G'), args)


@task
def neso_ensemble(config,
                  solver="views/gcc-hipsycl/bin/Electrostatic2D3V",
                  conditionsFileName="two_stream_conditions.xml",
                  meshFileName="two_stream_mesh.xml",
                  **args):

    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    env.script = 'neso'

    env.neso_solver = solver
    env.neso_conditions_file = conditionsFileName
    env.neso_mesh_file = meshFileName
    
    with_config(config)
    run_ensemble(config, sweep_dir, **args)
