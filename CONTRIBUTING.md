# Contributing to FabNESO

Thank you for considering contributing to FabNESO 🎉!

## Reporting problems

Please [create an issue](https://github.com/UCL/FabNESO/issues/new) if you experience any problems with using FabNESO or wish to request a new feature.

## Code and documentation contributions

We follow [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow) for making changes to the repository. All changes should be made on a new branch from the current tip of `main` (in a fork if you do not have write access to the repository), with the branch given a descriptive name. A pull request should be created from the new branch - if this is still a work in progress open or mark as a draft pull request. If the pull request addresses a specific existing issue [link the pull request to this issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).

Once you believe the changes are ready, request a review. The reviewer may approve the pull request directly or request changes are made. Any comments or suggestions from the reviewer should be addressed at this point.

GitHub Actions workflow jobs performing tests and checks will be automatically triggered on creation of a pull request and any subsequent pushes to the associated branch; the status of these checks will reported on the pull request and all checks will need to pass before the pull request can be merged.

Once all checks have passed, all review commments have been addressed and the reviewer(s) have approved the pull request, the pull request should be merged as a squash commit by a user with write access to the repository, and the associated branch deleted.

### Installing and using `pre-commit`

We use [`pre-commit`](https://pre-commit.com/) to run a standard set of Git hooks to automatically check commits for adherence with our [coding conventions](#coding-conventions). Once you have [`pre-commit` installed](https://pre-commit.com/#installation) run

```
pre-commit install
```

from the root of the repository to locally set up the Git hook scripts.

Once set up, the hooks will be run against the changes made in each commit with errors being reported (and the commit aborted) if checks fail. In some cases the scripts may be able to automatically fix the identified issues in which case it will be sufficient to simply stage the additional changes made and re-run the commit command.

### Coding conventions

All code should follow [the PEP8 style guide](https://peps.python.org/pep-0008/). Public facing functions (those not prefixed with an underscore `_`) should be annotated with type hints as described in [PEP 484](https://peps.python.org/pep-0484/).

As an exception to the above, the package name `FabNESO` does not follow [PEP8 guidelines on package and module names](https://peps.python.org/pep-0008/#package-and-module-names) as we instead have chosen to follow the FabSim3 convention of plugin names being in `CamelCase`. All other modules within the `FabNESO` package should be named according to PEP8 guidelines - that is all lower case names with underscores used as word separators when necessary to improve readability.

### Documentation conventions

All public facing functions should have a docstring following the [Google style format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) with types in the docstring omitted in favour of directly annotating the function with type hints.
