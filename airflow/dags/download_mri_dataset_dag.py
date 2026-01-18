from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os
import kagglehub
import shutil

DATA_DIR = "/opt/airflow/data/raw/mri_images"

def _download_mri_dataset():
    """Загрузка датасета МРТ головного мозга"""
    path = kagglehub.dataset_download(
        "fernando2rad/brain-tumor-mri-images-44c"
    )

    os.makedirs(DATA_DIR, exist_ok=True)

    for item in os.listdir(path):
        src = os.path.join(path, item) # формирование полного пути к папке-источнику
        dst = os.path.join(DATA_DIR, item) # полный путь, куда будет скопирован объект

        if os.path.isdir(src): # если текущий объект - директория
            shutil.copytree(src, dst, dirs_exist_ok=True) # копируем всю папку с содержимым
        else:
            shutil.copy2(src, dst)

with DAG(
    dag_id="download_brain_tumor_mri_dataset",
    start_date=days_ago(1),
    schedule_interval=None, # запуск только вручную
    catchup=False, # Airflow не будет запускать пропущенные интервалы за прошлое время
    tags=["dataset", "kaggle", "mri"]
) as dag:

    download_mri_dataset = PythonOperator( # оператор для скачивания датасета
        task_id="download_mri_dataset",
        python_callable=_download_mri_dataset,
    )

    notify = BashOperator( # оператор для информирования о кол-ве скачанных папок
        task_id="notify",
        bash_command=f'echo "Classes: $(ls {DATA_DIR} | wc -l)"',
    )

    download_mri_dataset >> notify