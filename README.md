<img style="width: 75px" src="https://github.com/social-ai-uoft/gem/blob/main/media/gem-pendant.png" />

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/social-ai-uoft/gem/main.svg)](https://results.pre-commit.ci/latest/github/social-ai-uoft/gem/main) ![pytest status](https://github.com/social-ai-uoft/gem/workflows/PyTest/badge.svg)

# gem

Gem is a general-purpose reinforcement learning engine that enables researchers, developers, and students to develop
and deploy reinforcement learning algorithms on new and pre-existing environments.

Gem can be regarded as an “operating system” for the RL process, in the sense that it unifies and abstracts foundational components from environment simulation all the way to policy search, exploration, and function approximation.

Our hope is that Gem will foster new research ideas, applications, and tools for a unified RL approach. We believe Gem will accelerate the rate of progress of the RL research space, as well as allow us to experiment with our own solutions to both novel and long-standing RL problems.

### NOTE: Gem is extremely experimental and subject to change!

## Development

**NOTE**: We recommend you follow these instructions in a fresh conda/virtual environment to keep packages isolated from other environments and/or Python versions. Python 3.9+ is required. If you have conda, run
```
conda create --name gem python=3.9
conda activate gem
```

### Getting started

Gem uses the [poetry](https://python-poetry.org/) package manager to manage its dependencies. Start by installing poetry by running ``pip install poetry``. To install gem in development mode, run the following command:
```
poetry install
```
in the folder containing the ``pyproject.toml`` file. 

See the [poetry](https://python-poetry.org/) documentation for more information and
installation instructions.


### Workflow Tools
We use a number of tools to simplify the development workflow for contributors. Among these tools include code formatters and linters (such as [autopep8](https://pypi.python.org/pypi/autopep8) and [flake8](https://pypi.python.org/pypi/flake8)) to report style errors and (try to) automatically format your code wherever possible, along with testing frameworks (such as [pytest](https://pypi.python.org/pypi/pytest)) to automatically test the code.

Though you may manually run these tools, we include a [pre-commit](https://pre-commit.com/) configuration to automatically run all CI tasks whenever you attempt to commit to the Gem repository. **We highly recommend you use pre-commit as pull requests will NOT be merged unless you pass ALL CI checks (including the pre-commit CI check).**

To setup pre-commit, start by confirming that its installed by running:
```
pre-commit --version
```
If this fails, you'll need to install pre-commit by running ``pip install pre-commit``. Then, run
```
pre-commit install
```
in the folder containing ``.pre-commit-config.yaml`` file. Now pre-commit will run automatically on every ``git commit``!

## Citing the project

TODO: Insert paper citation here

## Maintainers

Gem is currently maintained by [Shon Verch](https://github.com/galacticglum) (aka @galacticglum), [Wil Cunningham](https://www.psych.utoronto.ca/people/directories/all-faculty/william-cunningham) (aka [@wacunn](https://github.com/wacunn)), [Paul Stillman](https://www.paulstillman.com/) (aka [@paulstillman](https://github.com/paulstillman)), and [Ethan Jackson](https://github.com/ethancjackson) (aka @ethancjackson).

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email. If you have any questions, concerns, or suggestions, please post them on the [GitHub issues page](https://github.com/social-ai-uoft/gem/issues) or the [GitHub discussion page](https://github.com/social-ai-uoft/gem/discussions).

## Contributing

Gem is open source and is developed by a community of researchers, developers, and students. We welcome contributions from all levels of the community. To get started, please read the [contributing guide](CONTRIBUTING.md).

## Acknowledgments

TODO: Insert acknowledgements here
