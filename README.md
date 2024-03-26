[![SCM Compliance](https://scm-compliance-api.radix.equinor.com/repos/equinor/805e1799-b38a-48b5-a04f-2dff17ee744a/badge)](https://developer.equinor.com/governance/scm-policy/)

# Gordo core library.

# Table of Contents
* [Installation](#Installation)
* [Developers Instructions](#Developers-Instructions)
	* [Setup](#Setup)
	* [Pre-commit](#Pre-commit)
	* [Run tests](#Run-tests)
* [Contributing](#Contributing)

---

The main component can be found [here](https://github.com/equinor/gordo).

[Documentation is available on Read the Docs](https://gordo-core.readthedocs.io/)

## Installation

At least python 3.10 need to be installed in the system first.

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

Build documentation:
```console
> cd docs/
> poetry run make watch
```

## Contributing
We welcome contributions to this project! To get started, please follow these steps:

1. Fork this repository to your own GitHub account and then clone it to your local device.

```
git clone https://github.com/your-account/your-project.git
```

2. Create a new branch for your feature or bug fix.

```
git checkout -b your-feature-or-bugfix-branch
```

3. Make your changes and commit them with a descriptive message.

```
git commit -m "Add a new feature" -a
```

4. Push your changes to your forked repository.

```
git push origin your-feature-or-bugfix-branch
```

5. Open a pull request in this repository and describe the changes you made.

We'll review your changes and work with you to get them merged into the main branch of the project.