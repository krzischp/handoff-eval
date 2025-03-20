# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Default goal
.DEFAULT_GOAL := help

init: $(VENV)/bin/activate ## Create virtual environment and install dependencies (including handoff_eval in editable mode)

$(VENV)/bin/activate: requirements.txt requirements-dev.txt setup.py
	@echo "Creating virtual environment..."
	python -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "Installing handoff_eval in editable mode..."
	$(PIP) install -e .
	@echo "Virtual environment and dependencies are set up."

dev: ## Install only development dependencies
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt

activate: ## Activate the virtual environment
	@echo "Activating virtual environment..."
	@echo "Run the following command in your shell:"
	@echo ""
	@echo "    source $(VENV)/bin/activate"
	@echo ""

sync: ## Sync dependencies in an existing virtual environment
	@echo "Syncing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "Dependencies are up to date."

test: ## Run tests
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/

lint: ## Lint the code
	@echo "Linting code with flake8..."
	$(PYTHON) -m flake8 --ignore=E501,W503 handoff_eval tests

format: ## Auto-fix all code issues (unused imports, formatting, and sorting)
	@echo "Removing unused imports and variables with autoflake..."
	$(PYTHON) -m autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place handoff_eval tests
	@echo "Sorting imports with isort..."
	$(PYTHON) -m isort handoff_eval tests
	@echo "Formatting code with black..."
	$(PYTHON) -m black handoff_eval tests

clean: ## Clean temporary files
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-venv: ## Remove the virtual environment
	@echo "Removing virtual environment..."
	rm -rf $(VENV)

notebook: ## Run Jupyter Notebook with correct environment
	@echo "Starting Jupyter Notebook..."
	$(PYTHON) -m notebook

help: ## Help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
