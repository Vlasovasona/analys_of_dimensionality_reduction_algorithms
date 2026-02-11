import glob
import importlib.util
import os

import pytest
from airflow.models import DAG

DAG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "airflow/dags/**/*.py")
DAG_FILES = glob.glob(DAG_PATH, recursive=True)
DAG_FILES = [f for f in DAG_FILES if "scripts" not in f]

@pytest.mark.parametrize("dag_file", DAG_FILES)
def test_dag_integrity(dag_file):
    module_name, _ = os.path.splitext(dag_file)
    module_path = os.path.join(DAG_PATH, dag_file)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dag_objects = [
        var for var in vars(module).values()
        if isinstance(var, DAG)
    ]

    assert len(dag_objects) > 0, f"No DAG objects found in {dag_file}"