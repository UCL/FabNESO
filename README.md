# FabNESO

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/UCL/FabNESO/actions/workflows/tests.yml/badge.svg)](https://github.com/UCL/FabNESO/actions/workflows/tests.yml)
[![Linting](https://github.com/UCL/FabNESO/actions/workflows/linting.yml/badge.svg)](https://github.com/UCL/FabNESO/actions/workflows/linting.yml)
[![Documentation](https://github.com/UCL/FabNESO/actions/workflows/docs.yml/badge.svg)](https://github-pages.ucl.ac.uk/FabNESO/)
[![Licence](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](./LICENCE)

[Neptune Exploratory SOftware (NESO)](https://github.com/ExCALIBUR-NEPTUNE/NESO) plugin
for [FabSim3](https://github.com/djgroen/FabSim3), facilitating execution of NESO
simulations on both local and remote high performance computing systems via a unified
interface.

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

## About

### Project team

- Matt Graham ([matt-graham](https://github.com/matt-graham))
- Serge Guillas ([sguillas](https://github.com/sguillas))
- Tuomas Koskela ([tkoskela](https://github.com/tkoskela))
- Duncan Leggat ([dleggat](https://github.com/dleggat))
- Yiming Yang ([yyimingucl](https://github.com/yyimingucl))

### Research software engineering contact

Centre for Advanced Research Computing, University College London ([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations.@ucl.ac.uk))

## Getting started

### Prerequisites

You will need [FabSim3 to be installed](https://fabsim3.readthedocs.io/en/latest/installation/) on the machine you will use to dispatch simulation runs from.

[NESO must be installed](https://github.com/ExCALIBUR-NEPTUNE/NESO#building-with-spack) on the destination machine that will run the simulation.

### Installation

The plugin can be installed locally with FabSim3 by running:

```
fabsim localhost install_plugin:FabNESO
```

Before the code can be run, a file `machines_FabNESO_user.yml` must be created in the plugin's directory (`$FABSIM3_HOME/plugins/FabNESO/`) containing the paths to the built NESO binaries on each system that will be used for running.
An example file `machines_FabNESO_user_example.yml` is provided to use as a template.

### Running simulations locally

NESO runs by calling the desired solver with a conditions and mesh XML files that define the parameters of the simulation and geometry of the simulation domain respectively.
Examples of these configuration files are provided in the directories `config_files/two_stream/` (intended for use with NESO's `Electrostatic2D3V` solver) and `config_files/2Din3D-hw/` (for use with the `H3LAPD` solver).

#### Running a single simulation

The FabSIM `neso` task runs a single simulation on the machine of your chosing.
To run NESO locally using the `Electrostatic2D3V` solver with [the `two_stream` example](https://github.com/ExCALIBUR-NEPTUNE/NESO/tree/main/examples/Electrostatic2D3V/two_stream), run the following command:

```
fabsim localhost neso:two_stream
```

The first positional argument after `neso:` specifies the directory within `config_files` that holds the conditions and mesh files to be used.

Additional arguments that can be given the to the `neso` task include:

- `solver` : chose which NESO solver to run (default = `Electrostatic2D3V`),
- `conditions_file_name` : Name of conditions XML file in configuration directory (default = `conditions.xml`),
- `mesh_file_name` : Name of mesh XML file in configuration directory (default = `mesh.xml`).

Any additional keyword arguments to the task will be used to override the value of parameters in the conditions file. For example to run the `two_stream` example with the `Electorstatic2D3V` solver, with the `num_particles_total` parameter overridden to be 10000 run

```
fabsim localhost neso:two_stream,num_particles_total=10000
```

To run the H3LAPD solver with [the `2Din3D-hw` example configuration](https://github.com/ExCALIBUR-NEPTUNE/NESO/tree/main/examples/H3LAPD/2Din3D-hw), for example, the following command should be run:

```
fabsim localhost neso:2Din3D-hw,solver=H3LAPD
```

To retrieve the results of a job, run the command:

```
fabsim localhost fetch_results
```

#### Running an ensemble of simulations

The FabSIM `neso_ensemble` task runs a series of FabSIM jobs taking a `SWEEP` directory as its primary input.
This `SWEEP` directory contains any number of subdirectories, each containing a conditions and mesh file for individual NESO jobs.

A utility script [`FabNESO/make_sweep_dir.py`](https://github.com/UCL/FabNESO/blob/main/FabNESO/make_sweep_dir.py) is provided to automatically build this sweep directory and encode the input conditions files with templated parameters selected by the user. It should be executed as `python -m FabNESO.make_sweep_dir`.

The script takes the following input parameters (see also output of passing `--help` option):

- `--sweep_path` : a path that will act as the SWEEP directory (default = `$FABSIM3_HOME/plugins/FabNESO/config_files/two_stream_ensemble`),
- `--n_dirs` : Number of divisions in grid for each parameter (default = `5`),
- `--destructive` : Deletes the previous tree if it already exists (default = `False`),
- `--copy_dir` : Copy contents of this dir to each sweep dir (default = `$FABSIM3_HOME/plugins/FabNESO/config_files/two_stream`),
- `--edit_file`: Template a parameter in this file (default = `conditions.xml`).

To template a single parameter, the following three command line arguments should be added:

- `--para_to_template` : The name of the parameter to template (default = `""`),
- `--scan_min` : The minimum value of the parameter scan (default = `0`),
- `--scan_max` : The maximum value of the parameter scan (default = `0`).

An example use to template the `particle_initial_velocity` parameter for 4 values between 0.1 and 2.0 using the the two_stream example files would therefore be:

```
python -m FabNESO.make_sweep_dir --para_to_template="particle_initial_velocity" --scan_min=0.1 --scan_max=2.0 --n_dirs=4
```

The script can also template an arbitrary number of parameters in the conditions file using the following command line argument:

- `--parameter_dict` : A Python dict of the parameters to be scanned, with associated minimum and maximum values as a list (default = `""`)

To template both the above `particle_initial_velocity` and the `particle_charge_density` of the simulation between 102 and 109, run the following command:

```
python -m FabNESO.make_sweep_dir --parameter_dict="{'particle_initial_velocity': [0.1,2.0], 'particle_charge_density': [102,109]}" --n_dirs=4
```

This will create 16 directories in the `config_files/two_stream_ensemble` for the combination of these scans.

The NESO task `neso_ensemble` can then be run over this sweep directory using the command:

```
fabsim localhost neso_ensemble:two_stream_ensemble
```

`neso_ensemble` takes the additional input parameters:

- `solver` : chose which NESO solver to run (default = `Electrostatic2D3V`),
- `conditions_file_name` : Name of conditions XML file in SWEEP directories (default = `conditions.xml`),
- `mesh_file_name` : Name of mesh XML file in the SWEEP directories (default = `mesh.xml`).

The results of the jobs are recovered using the same fetch command:

```
fabsim localhost fetch_results
```

### Running simulations on a remote system

To run FabNESO on a remote machine, the previous instructions for running locally can be followed with `localhost` replaced with the remote system of choice.
Running a single NESO simulation on Kathleen, for example, can be carried out with the command:

```
fabsim Kathleen neso:two_stream
```

A list of possible remote destinations for FabSIM is provided using the command:

```
fabsim -l machines
```

**Note** that NESO must be installed on the remote machine, and that the `bin/` directory of said NESO installation must be added to the plugin's `machines_FabNESO_user.yml` file for each remote system.

Further information on running FabSIM can be found in the [FabSIM documentation](https://fabsim3.readthedocs.io/en/latest/).

## Contributing

If you are interested in contributing to FabNESO please see our separate [contributors guide](CONTRIBUTING.md).

## Acknowledgements

This work was funded by a grant from the ExCALIBUR programme.
