# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Default goal
.DEFAULT_GOAL := help

## Create virtual environment and install dependencies (including handoff_eval in editable mode)
init: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt setup.py
	@echo "Creating virtual environment..."
	python -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Installing handoff_eval in editable mode..."
	$(PIP) install -e .
	@echo "Virtual environment and dependencies are set up."

## Run tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/

## Lint the code
lint:
	@echo "Linting code with flake8..."
	$(PYTHON) -m flake8 handoff_eval tests

## Format the code with black
format:
	@echo "Formatting code with black..."
	$(PYTHON) -m black handoff_eval tests

## Clean temporary files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

## Remove the virtual environment
clean-venv:
	@echo "Removing virtual environment..."
	rm -rf $(VENV)

## Run Jupyter Notebook with correct environment
notebook:
	@echo "Starting Jupyter Notebook..."
	$(PYTHON) -m notebook

## Help message
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
