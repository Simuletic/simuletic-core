# simuletic-core

**simuletic-core is the open-source core Python toolkit for Simuletic.**

Simuletic is focused on synthetic-to-real computer vision workflows: training on synthetic datasets, validating against real-world data, and measuring whether synthetic data improves real-world model performance.

> Synthetic data is only useful if it improves real-world model performance.

## Current status

This repository is early and intentionally lightweight. The current package provides the initial Python project structure, test setup, CI, and a minimal CLI.

Implemented today:

* Python package import name: `simuletic`
* CLI command: `simuletic`
* Version command: `simuletic version`
* Basic tests, linting, typing configuration, and GitHub Actions CI

Not implemented yet:

* training workflows
* evaluation workflows
* inference workflows
* dataset validation
* benchmark reports
* API clients
* ML framework integrations

## Installation

Install locally with pip:

```bash
pip install .
```

Or with uv:

```bash
uv pip install .
```

For editable development:

```bash
pip install -e ".[dev]"
```

or:

```bash
uv pip install -e ".[dev]"
```

## Quickstart

Check that the CLI is installed:

```bash
simuletic version
```

Run the test suite and checks:

```bash
pytest
ruff check .
mypy src
```

## Repository scope

The core package should stay lightweight. Heavy computer vision and ML dependencies are intentionally not included in the initial package.

Future functionality will be added carefully around professional synthetic-to-real computer vision workflows, including training, real-world evaluation, inference, benchmarking, and optional integrations.
