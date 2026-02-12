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
	pip install -r requirements-test.txt
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