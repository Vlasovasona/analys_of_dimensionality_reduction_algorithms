.PHONY: help install lint test clean docker-up docker-down

help:
	@echo "Доступные команды:"
	@echo "  make install     - Установка зависимостей для разработки"
	@echo "  make lint        - Запуск линтеров (flake8, black)"
	@echo "  make test        - Запуск тестов"
	@echo "  make clean       - Очистка кэша"
	@echo "  make docker-up   - Запуск Docker контейнеров"
	@echo "  make docker-down - Остановка Docker контейнеров"

install:
	pip install --upgrade pip

	# 1. Устанавливаем Airflow и всё для DAG
	pip install apache-airflow==2.7.3 \
	    numpy==1.24.3 \
	    pandas==2.0.3 \
	    scikit-learn==1.3.2 \
	    boto3==1.34.0 \
	    pytest==7.4.3 \
	    pytest-cov==4.1.0 \
	    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-3.10.txt"

	# 2. Устанавливаем инструменты разработчика (без constraints)
	pip install flake8==6.1.0 \
	            black==23.11.0 \
	            pylint==3.0.2

	pip install -e .

lint:
	@echo "=== Запуск Flake8 ==="
	flake8 airflow/ scripts/ tests/
	@echo "=== Запуск Black (check only) ==="
	black --check airflow/ scripts/ tests/
	@echo "=== Запуск Pylint ==="
	pylint --fail-under=8.0 --exit-zero airflow/ scripts/ tests/

test:
	@echo "=== Запуск тестов ==="
	PYTHONPATH=${PWD} pytest tests/ -v --tb=short --cov=./ --cov-report=term-missing

test-dag:
	@echo "=== Тестирование DAG ==="
	PYTHONPATH=${PWD} pytest tests/airflow/dags/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

format:
	black airflow/ scripts/ tests/