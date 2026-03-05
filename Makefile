SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG ?= config.yaml
RUNS ?= 10

DOWNLOADS_ROOT ?= thessaloniki_govgr/downloads
CALIBRATION_YEAR ?= 2025
VALIDATION_YEAR ?= 2026
TARGETS_DIR ?= thessaloniki_govgr/targets/post_metro_2025_2026

.PHONY: help venv install install-dev run run-batch run-postmetro50 assess \
	fetch-realtime fetch-historical-2025 fetch-historical-2026 build-targets \
	test smoke-govgr clean-pycache

help:
	@echo "SAS operator commands"
	@echo ""
	@echo "  make install                 Create .venv and install package"
	@echo "  make install-dev             Install package + dev tools"
	@echo "  make run                     Run one simulation (CONFIG=config.yaml)"
	@echo "  make run-batch RUNS=10       Run batch simulations"
	@echo "  make run-postmetro50         Run using 50 km/h-capped Thessaloniki network"
	@echo "  make assess                  Run resilience assessment"
	@echo "  make fetch-realtime          Download realtime govgr datasets"
	@echo "  make fetch-historical-2025   Download 2025 historical speed + travel times"
	@echo "  make fetch-historical-2026   Download 2026 historical speed + travel times"
	@echo "  make build-targets           Build calibration/validation target CSVs"
	@echo "  make test                    Run all tests"

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

run-postmetro50:
	$(PYTHON) runner.py --config config_thessaloniki_postmetro_50kph.yaml

assess:
	$(PYTHON) resilience_assessment.py --config $(CONFIG)

fetch-realtime:
	$(PYTHON) govgr_downloader.py --source realtime --dataset all --output-dir $(DOWNLOADS_ROOT)/realtime_latest

fetch-historical-2025:
	$(PYTHON) govgr_downloader.py --source historical --dataset speed --historical-pattern _2025 --output-dir $(DOWNLOADS_ROOT)/historical_2025
	$(PYTHON) govgr_downloader.py --source historical --dataset travel_times --historical-pattern _2025 --output-dir $(DOWNLOADS_ROOT)/historical_2025

fetch-historical-2026:
	$(PYTHON) govgr_downloader.py --source historical --dataset speed --historical-pattern _2026 --output-dir $(DOWNLOADS_ROOT)/historical_2026
	$(PYTHON) govgr_downloader.py --source historical --dataset travel_times --historical-pattern _2026 --output-dir $(DOWNLOADS_ROOT)/historical_2026

build-targets:
	$(PYTHON) govgr_targets.py \
		--downloads-root $(DOWNLOADS_ROOT) \
		--calibration-year $(CALIBRATION_YEAR) \
		--validation-year $(VALIDATION_YEAR) \
		--output-dir $(TARGETS_DIR)

test:
	$(PYTHON) -m pytest -q tests

smoke-govgr:
	$(PYTHON) -m pytest -q tests/test_govgr_downloader.py

clean-pycache:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
