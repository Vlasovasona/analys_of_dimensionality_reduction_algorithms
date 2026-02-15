import os
import shutil
import numpy as np
import io
import json
import kagglehub
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

BUCKET_NAME = "mri-dataset"
DATA_DIR = "/opt/airflow/data/raw/mri_images"
IMG_SIZE = 64
RAW_PREFIX = "mri/raw/"
PROCESSED_PREFIX = "mri/processed/"
BATCH_SIZE = 128
TMP_DIR = "/tmp/mri_batches"


def _download_mri_dataset():
    """Скачивание датасета из Kaggle в локальное хранилище"""
    try:
        path = kagglehub.dataset_download("fernando2rad/brain-tumor-mri-images-44c")
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")

    os.makedirs(DATA_DIR, exist_ok=True)  # создает целевой каталог
    for item in os.listdir(path):  # берет все содержимое скачанного датасета (item - имя элемента, не полный путь)
        src = os.path.join(path, item)  # формирование полного пути
        dst = os.path.join(DATA_DIR, item)  # формирование пути назначения
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)  # копирует всю папку целиком
        else:
            shutil.copy2(src, dst)  # копирует один файл


def _upload_images_to_s3():
    """Загрузка сырых изображений в S3"""
    s3 = S3Hook(aws_conn_id="s3")  # хук получает данные подключения "s3"
    for root, _, files in os.walk(DATA_DIR):  # кортеж (папка, список подпапок, список файлов)
        for file in files:
            local_path = os.path.join(root, file)  # формирование абсолютного пути к файлу
            s3_key = local_path.replace(DATA_DIR, RAW_PREFIX)  # путь s3
            s3.load_file(filename=local_path, key=s3_key, bucket_name=BUCKET_NAME, replace=True)


def _save_batch(X, y, batch_id, s3):
    """Сохраняет батч изображений и меток в S3"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    X_path = f"{TMP_DIR}/X_{batch_id:04d}.npy"
    y_path = f"{TMP_DIR}/y_{batch_id:04d}.npy"

    np.save(X_path, X)
    np.save(y_path, y)

    s3.load_file(filename=X_path, key=f"{PROCESSED_PREFIX}X_{batch_id:04d}.npy", bucket_name=BUCKET_NAME, replace=True)
    s3.load_file(filename=y_path, key=f"{PROCESSED_PREFIX}y_{batch_id:04d}.npy", bucket_name=BUCKET_NAME, replace=True)

    os.remove(X_path)
    os.remove(y_path)


def _preprocess_mri_images():
    """Нормализация, resize картинок, формирование батчей, загрузка обработанных данных в S3"""
    import cv2

    os.makedirs(TMP_DIR, exist_ok=True)
    s3 = S3Hook(aws_conn_id="s3")

    keys = [k for k in s3.list_keys(BUCKET_NAME, RAW_PREFIX) if k.lower().endswith((".jpg", ".png", ".jpeg"))]
    class_names = sorted({k.split("/")[2] for k in keys})
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    batch_X, batch_y = [], []
    batch_id = 0

    for idx, key in enumerate(keys):
        try:
            obj = s3.get_key(key=key, bucket_name=BUCKET_NAME)
            buf = io.BytesIO()
            obj.download_fileobj(buf)
            buf.seek(0)

            img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
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
            print(f"Ошибка обработки {key}: {e}")

    if batch_X:
        _save_batch(batch_X, batch_y, batch_id, s3)

    s3.load_string(
        string_data=json.dumps(class_to_idx),
        key=f"{PROCESSED_PREFIX}class_to_idx.json",
        bucket_name=BUCKET_NAME,
        replace=True,
    )
