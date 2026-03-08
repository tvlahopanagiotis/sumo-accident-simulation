SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG ?= config.yaml
RUNS ?= 10

.PHONY: help venv install install-dev run run-batch test clean-pycache

help:
	@echo "SAS — SUMO Accident Simulation"
	@echo ""
	@echo "  make install             Create .venv and install package"
	@echo "  make install-dev         Install package + dev tools"
	@echo "  make run                 Run one simulation (CONFIG=config.yaml)"
	@echo "  make run-batch RUNS=10   Run batch simulations"
	@echo "  make test                Run all tests"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -e ".[dev]"

run:
	$(PYTHON) runner.py --config $(CONFIG)

run-batch:
	$(PYTHON) runner.py --config $(CONFIG) --runs $(RUNS)

test:
	$(PYTHON) -m pytest -q tests

clean-pycache:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
