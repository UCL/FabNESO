# FabNESO

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status](https://github.com/UCL/FabNESO/actions/workflows/tests.yml/badge.svg)](https://github.com/UCL/FabNESO/actions/workflows/tests.yml)
[![Linting status](https://github.com/UCL/FabNESO/actions/workflows/linting.yml/badge.svg)](https://github.com/UCL/FabNESO/actions/workflows/linting.yml)
[![Documentation status](https://github.com/UCL/FabNESO/actions/workflows/docs.yml/badge.svg)](https://github.com/UCL/FabNESO/actions/workflows/docs.yml)
[![Licence](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](./LICENCE)
[![Documentation](https://img.shields.io/badge/Sphinx-documentation-blue?logo=sphinx&logoColor=white)](https://github-pages.ucl.ac.uk/FabNESO/)

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
An example file [`machines_FabNESO_user_example.yml`](machines_FabNESO_user_example.yml) is provided to use as a template.

### Running tasks locally

NESO runs by calling the desired solver with a conditions and mesh XML files that define the parameters of the simulation and geometry of the simulation domain respectively.
Examples of these configuration files are provided in the directories `config_files/two_stream/` (intended for use with NESO's `Electrostatic2D3V` solver) and `config_files/2Din3D-hw/` (for use with the `H3LAPD` solver).

#### Running a single simulation

The [FabNESO `neso` task](http://github-pages.ucl.ac.uk/FabNESO/FabNESO.html#FabNESO.neso) runs a single simulation on the machine of your choosing.
To run NESO locally using the `Electrostatic2D3V` solver with [the `two_stream` example](https://github.com/ExCALIBUR-NEPTUNE/NESO/tree/main/examples/Electrostatic2D3V/two_stream), run the following command:

```
fabsim localhost neso:two_stream
```

The first positional argument after `neso:` specifies the directory within `config_files` that holds the conditions and mesh files to be used.

Additional arguments that can be given the to the `neso` task include:

- `solver` : chose which NESO solver to run (default = `Electrostatic2D3V`),
- `conditions_file_name` : Name of conditions XML file in configuration directory (default = `conditions.xml`),
- `mesh_file_name` : Name of mesh XML file in configuration directory (default = `mesh.xml`).

A full list of the arguments that can be passed to the task is available in the [package documentation](http://github-pages.ucl.ac.uk/FabNESO/FabNESO.html#FabNESO.neso).

The `neso` task also supports passing additional keyword arguments to override the value of parameters in the conditions file. For example to run the `two_stream` example with the `Electrostatic2D3V` solver, with the `num_particles_total` parameter overridden to be 10000 run

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

FabNESO also provides tasks for running ensembles of NESO simulations - the links to the package documentation for the tasks below give details of the arguments that can be passed.

- [`neso_grid_ensemble` task](http://github-pages.ucl.ac.uk/FabNESO/FabNESO.tasks.html#FabNESO.tasks.neso_grid_ensemble): Run an ensemble of NESO solver instances on a parameter grid formed of the tensor product of evenly spaced grids on each parameter.
- [`neso_qmc_ensemble` task](http://github-pages.ucl.ac.uk/FabNESO/FabNESO.tasks.html#FabNESO.tasks.neso_qmc_ensemble): Run an ensemble of NESO solver instances on quasi-random parameter samples from a uniform distribution over a product of intervals (hypercube) in parameter space, generated using [Chaospy](https://chaospy.readthedocs.io/en/master/). This allows [quasi Monte Carlo (QMC)](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method) estimates of integrals to be computed.
- [`neso_pce_ensemble` task](http://github-pages.ucl.ac.uk/FabNESO/FabNESO.tasks.html#FabNESO.tasks.neso_pce_ensemble): Run an ensemble of NESO solver instances on parameter values forming the nodes of a quadrature rule over a product of intervals (hypercube) in parameter space. This allows performing a [polynomial chaos expansion (PCE)](https://en.wikipedia.org/wiki/Polynomial_chaos) of the solver outputs using [Chaospy](https://chaospy.readthedocs.io/en/master/), with the outputs approximated by an expansion in a set of orthogonal (with respect to the assumed uniform distribution over the parameter space) polynomials. The outputs of this task can be analysed using the [`neso_pce_analysis` task](http://github-pages.ucl.ac.uk/FabNESO/FabNESO.tasks.html#FabNESO.tasks.neso_pce_analysis) to form a PCE approximation to the model.

#### Calibrating a model using PyVBMC

FabNESO also provides [a task `neso_vbmc`](http://github-pages.ucl.ac.uk/FabNESO/FabNESO.tasks.html#FabNESO.tasks.neso_vbmc) for calibrating (inferring the posterior distribution on) the parameters of a NESO model given data corresponding to observations of the model output and a distribution over the parameters corresponding to our prior beliefs. This uses the [PyVBMC](https://acerbilab.github.io/pyvbmc/) package which provides an implementation of [_variational Bayesian Monte Carlo_ (Acerbi, 2018)](https://papers.nips.cc/paper/2018/hash/747c1bcceb6109a4ef936bc70cfe67de-Abstract.html), an approximate inference method designed for fitting computationally expensive models with a limited budget of model evaluations.

### Running tasks on a remote system

To run FabNESO on a remote machine, the previous instructions for running locally can be followed with `localhost` replaced with the remote system of choice.
Running a single NESO simulation on ARCHER2, for example, can be carried out with the command:

```
fabsim archer2 neso:two_stream
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
