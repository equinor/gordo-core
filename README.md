# Gordo core library.

The main component can be found [here](https://github.com/equinor/gordo).

[Documentation is available on Read the Docs](https://gordo-core.readthedocs.io/)

## Installation

At least python 3.9 need to be installed in the system first.

```
pip3 install gordo-core
```

## Developers Instructions

### Setup

Install [poetry](https://python-poetry.org/docs/#installation).

Setup and run development shell instance:

```console
> poetry install
> poetry shell
```
### Pre-commit

You could also install and apply [pre-commit](https://pre-commit.com/#usage) hooks.

### Run tests

Install [docker](https://docs.docker.com/engine/install/) (or similar container manager) if you want to run test-suite.

Run tests (except docker-related ones):

```console
> poetry run pytest -n auto -m "not dockertest"
```

Run docker-related tests:
```console
> poetry run pytest -m "dockertest"
```
