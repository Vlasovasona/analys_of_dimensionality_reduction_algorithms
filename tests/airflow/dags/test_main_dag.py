import glob
import importlib.util
import os
import sys
import pytest
from airflow.models import DAG
from airflow.exceptions import AirflowDagCycleException

# поднимаемся на три уровня вверх от директории расположения текущесго файла
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# рекурсивно обходим все файлы .py директории с дагами
DAG_PATH = os.path.join(PROJECT_ROOT, "airflow/dags/**/*.py")
DAG_FILES = glob.glob(DAG_PATH, recursive=True)

# запускает тест отдельно для каждого файла из DAG_FILES, dag_file - параметр, в который подставляется путь к файлу
@pytest.mark.parametrize("dag_file", DAG_FILES)
def test_dag_integrity(dag_file):
    """Тест на благонадежность DAG, отсутствие циклов"""
    module_name = os.path.splitext(os.path.basename(dag_file))[0] # берем название файла без расширения
    module_path = dag_file

    # загрузка модуля module_name из файла по пути module_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dag_objects = [
        var for var in vars(module).values()
        if isinstance(var, DAG)
    ]

    assert len(dag_objects) > 0, f"В файле {dag_file} не найдено DAG"

    for dag in dag_objects:
        # проверка на циклы через topological_sort
        try:
            dag.topological_sort()
        except AirflowDagCycleException as e:
            pytest.fail(f"Найден цикл в DAG {dag.dag_id}: {e}")

        assert dag.start_date is not None, f"DAG {dag.dag_id} не имеет start_date"
        assert len(dag.tasks) > 0, f"DAG {dag.dag_id} не содержит задач"