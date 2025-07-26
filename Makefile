# Makefile for OCR Table Extraction Pipeline
# Provides common development tasks and shortcuts

.PHONY: help install install-dev test test-fast test-cov lint format type-check clean build docs serve-docs release pre-commit setup-dev

# Default target
help:
	@echo "OCR Table Extraction Pipeline - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install with development dependencies"
	@echo "  setup-dev     Complete development setup (install + pre-commit)"
	@echo ""
	@echo "Development:"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run all linters (ruff, flake8, bandit)"
	@echo "  type-check    Run mypy type checking"
	@echo "  pre-commit    Run all pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run tests without slow integration tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Clean build artifacts and cache"
	@echo "  build         Build distribution packages"
	@echo "  release       Build and upload to PyPI (requires auth)"

# Python and pip commands
PYTHON := python
PIP := pip
POETRY := poetry

# Determine if we should use poetry or pip
USE_POETRY := $(shell command -v poetry 2> /dev/null)

# Installation targets
install:
ifdef USE_POETRY
	$(POETRY) install
else
	$(PIP) install -e .
endif

install-dev:
ifdef USE_POETRY
	$(POETRY) install --with dev,test,docs
else
	$(PIP) install -e ".[dev,test,docs]"
endif

setup-dev: install-dev
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "Development environment setup complete!"

# Code formatting
format:
	black src/ tests/
	isort src/ tests/
	@echo "Code formatting complete!"

# Linting
lint:
	ruff check src/ tests/
	flake8 src/ tests/
	bandit -r src/ -c pyproject.toml
	@echo "Linting complete!"

# Type checking
type-check:
	mypy src/

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Testing
test:
ifdef USE_POETRY
	$(POETRY) run pytest
else
	pytest
endif

test-fast:
ifdef USE_POETRY
	$(POETRY) run pytest -m "not slow"
else
	pytest -m "not slow"
endif

test-cov:
ifdef USE_POETRY
	$(POETRY) run pytest --cov=ocr_pipeline --cov-report=html --cov-report=term-missing
else
	pytest --cov=ocr_pipeline --cov-report=html --cov-report=term-missing
endif

# Documentation
docs:
ifdef USE_POETRY
	$(POETRY) run sphinx-build -b html docs/ docs/_build/html
else
	sphinx-build -b html docs/ docs/_build/html
endif

serve-docs: docs
ifdef USE_POETRY
	$(POETRY) run python -m http.server 8000 --directory docs/_build/html
else
	$(PYTHON) -m http.server 8000 --directory docs/_build/html
endif

# Build and release
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "Cleanup complete!"

build: clean
ifdef USE_POETRY
	$(POETRY) build
else
	$(PYTHON) -m build
endif

release: build
ifdef USE_POETRY
	$(POETRY) publish
else
	$(PYTHON) -m twine upload dist/*
endif

# Development shortcuts
dev-check: format lint type-check test-fast
	@echo "Development checks complete!"

ci-check: lint type-check test-cov
	@echo "CI checks complete!"

# Pipeline execution shortcuts
stage1:
ifdef USE_POETRY
	$(POETRY) run ocr-stage1
else
	ocr-stage1
endif

stage2:
ifdef USE_POETRY
	$(POETRY) run ocr-stage2
else
	ocr-stage2
endif

pipeline: stage1 stage2
	@echo "Complete pipeline execution finished!"

# Example and config generation
examples:
ifdef USE_POETRY
	$(POETRY) run python -c "from ocr_pipeline.config.loader import export_example_configs; export_example_configs('examples/configs')"
else
	$(PYTHON) -c "from ocr_pipeline.config.loader import export_example_configs; export_example_configs('examples/configs')"
endif
	@echo "Example configurations exported to examples/configs/"

# Docker targets (if Dockerfile exists)
docker-build:
	docker build -t ocr-pipeline:latest .

docker-run:
	docker run --rm -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output ocr-pipeline:latest

# Performance profiling
profile:
ifdef USE_POETRY
	$(POETRY) run python -m cProfile -o profile_output.prof -c "import ocr_pipeline; # Add profiling code here"
else
	$(PYTHON) -m cProfile -o profile_output.prof -c "import ocr_pipeline; # Add profiling code here"
endif

# Security audit
security:
	bandit -r src/ -c pyproject.toml
	safety check
	@echo "Security audit complete!"

# Dependency updates
update-deps:
ifdef USE_POETRY
	$(POETRY) update
else
	$(PIP) install --upgrade -r requirements-dev.txt
endif

# Quick development workflow
quick: format lint test-fast
	@echo "Quick development check complete!"