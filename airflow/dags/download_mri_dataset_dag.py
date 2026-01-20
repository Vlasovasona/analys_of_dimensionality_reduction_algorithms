from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import numpy as np
import os
import kagglehub
import shutil
import cv2
import json
import io


BUCKET_NAME = "mri-dataset"
DATA_DIR = "/opt/airflow/data/raw/mri_images"

IMG_SIZE = 64

RAW_PREFIX = "mri/raw/"
PROCESSED_PREFIX = "mri/processed/"

BATCH_SIZE = 128
TMP_DIR = "/tmp/mri_batches"


def _download_mri_dataset():
    """Загрузка датасета МРТ головного мозга из Kaggle в локальное хранилище"""
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


def _upload_images_to_s3():
    """Загружает необработанные изображения из локального хранилища в бакет S3."""

    s3 = S3Hook(aws_conn_id="s3")

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            local_path = os.path.join(root, file)

            # Формируем ключ в бакете
            s3_key = local_path.replace(
                DATA_DIR,
                "mri/raw"
            )

            s3.load_file(
                filename=local_path,
                key=s3_key,
                bucket_name=BUCKET_NAME,
                replace=True
            )

def _save_batch(X, y, batch_id, s3):
    """
    Сохраняет пакет изображений и меток в локальные .npy файлы и загружает их в S3, после чего удаляет локальные файлы.

    params:
    ----------
    X : array-like
        Массив изображений, который нужно сохранить (будет приведён к np.float32)
    y : array-like
        Массив меток/классов (будет приведён к np.int64)
    batch_id : int
        Номер текущего батча. Используется для именования файлов (X_0001.npy, y_0001.npy)
    s3 : S3Hook
        Объект S3Hook для загрузки файлов в указанный S3-bucket
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    X_path = f"{TMP_DIR}/X_{batch_id:04d}.npy"
    y_path = f"{TMP_DIR}/y_{batch_id:04d}.npy"

    np.save(X_path, X)
    np.save(y_path, y)

    s3.load_file(
        filename=X_path,
        key=f"{PROCESSED_PREFIX}X_{batch_id:04d}.npy",
        bucket_name=BUCKET_NAME,
        replace=True
    )

    s3.load_file(
        filename=y_path,
        key=f"{PROCESSED_PREFIX}y_{batch_id:04d}.npy",
        bucket_name=BUCKET_NAME,
        replace=True
    )
    os.remove(X_path)
    os.remove(y_path)


def _preprocess_mri_images():
    """Нормализация, resize картинок, формирование батчей, загрузка обработанных данных в S3"""

    os.makedirs(TMP_DIR, exist_ok=True)
    s3 = S3Hook(aws_conn_id="s3")

    keys = [
        k for k in s3.list_keys(BUCKET_NAME, RAW_PREFIX)
        if k.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    class_names = sorted({k.split("/")[2] for k in keys})
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    batch_X, batch_y = [], []
    batch_id = 0

    for idx, key in enumerate(keys):
        try:
            obj = s3.get_key(
                key=key,
                bucket_name=BUCKET_NAME
            )
            buf = io.BytesIO()
            obj.download_fileobj(buf)
            buf.seek(0)

            img = cv2.imdecode(
                np.frombuffer(buf.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0

            batch_X.append(img)
            batch_y.append(class_to_idx[key.split("/")[2]])

            if len(batch_X) == BATCH_SIZE:
                _save_batch(batch_X, batch_y, batch_id, s3)
                batch_X, batch_y = [], []
                batch_id += 1

        except Exception as e:
            print(f"ERROR processing {key}: {e}")

    # последний неполный батч
    if batch_X:
        _save_batch(batch_X, batch_y, batch_id, s3)

    # сохранить словарь меток
    s3.load_string(
        string_data=json.dumps(class_to_idx),
        key=f"{PROCESSED_PREFIX}class_to_idx.json",
        bucket_name=BUCKET_NAME,
        replace=True
    )

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

    upload_images_to_s3 = PythonOperator( # оператор для загрузки необработанных данных в хранилище S3
        task_id="upload_images_to_s3",
        python_callable=_upload_images_to_s3,
    )

    preprocess_mri_images = PythonOperator( # оператор для обработки "сырых" картинок и загрузки в S3
        task_id="preprocess_mri_images",
        python_callable=_preprocess_mri_images,
    )

    download_mri_dataset >> upload_images_to_s3 >> preprocess_mri_images
