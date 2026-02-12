.PHONY: help install lint test clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make lint       - Run linters"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean cache files"

install:
	pip install --upgrade pip
	pip install -e .
	pip install apache-airflow pytest pytest-cov flake8 black pylint

lint:
	@echo "Running Flake8..."
	flake8 airflow/ scripts/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "Running Black..."
	black --check airflow/ scripts/ tests/ || true
	@echo "Running Pylint..."
	pylint --fail-under=8.0 --exit-zero airflow/ scripts/ tests/

test:
	export PYTHONPATH=${PWD} && \
	pytest tests/ -v --cov=./ --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +