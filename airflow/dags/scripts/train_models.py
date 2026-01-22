import os
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


BUCKET_NAME = "mri-dataset"
PROCESSED_PREFIX = "mri/processed/"
LOCAL_DATA_DIR = "/tmp/mri_train_data"


def load_data_from_s3():
    """
    Загружает батчи из S3 в память
    """
    s3 = S3Hook(aws_conn_id="s3")
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    keys = s3.list_keys(BUCKET_NAME, PROCESSED_PREFIX) # получили список всех файлов внутри mri/processed/

    X_list, y_list = [], []

    for key in keys: # цикл по файлам в S3
        if key.endswith(".npy") and "X_" in key: # если это один из батчей X
            local_x = Path(LOCAL_DATA_DIR) / os.path.basename(key)
            local_y = Path(LOCAL_DATA_DIR) / os.path.basename(key.replace("X_", "y_"))

            # создаём директорию, если её нет
            local_x.parent.mkdir(parents=True, exist_ok=True)
            local_y.parent.mkdir(parents=True, exist_ok=True)

            # скачиваем батчи с данными для обучения и метками классов
            s3.get_conn().download_file(
                Bucket=BUCKET_NAME,
                Key=key,
                Filename=str(local_x)
            )
            s3.get_conn().download_file(
                Bucket=BUCKET_NAME,
                Key=key.replace("X_", "y_"),
                Filename=str(local_y)
            )

            X = np.load(local_x)
            y = np.load(local_y)

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list) # склеиваем все батчи
    y = np.concatenate(y_list)

    return X.reshape(len(X), -1), y # разворачиваем картинки в векторы


def _train_model(
    model_type: str,
    pca_components: int = 128,
    mlflow_uri: str = "http://mlflow:5000",
):
    """
    Универсальная функция обучения модели для Airflow
    """
    mlflow.set_tracking_uri("http://mlflow:5000") # куда отправлять логи
    mlflow.set_experiment("mri-brain-tumor") # все эксперименты идут в один namespace

    with mlflow.start_run(run_name=model_type): # один запуск - одна модель
        X, y = load_data_from_s3()

        mlflow.log_param("model_type", model_type) # логируем параметры обучения
        mlflow.log_param("original_dim", X.shape[1])
        mlflow.log_param("pca_components", pca_components)

        pca = PCA(n_components=pca_components) # инициализируем pca
        X_pca = pca.fit_transform(X) # обучаем и применяем pca

        # создаем модели в зависимости от значения аргумента model_type
        if model_type == "logreg":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "svm":
            model = SVC(kernel="rbf")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_pca, y)
        preds = model.predict(X_pca)

        acc = accuracy_score(y, preds) # логируем метрики качества классификации
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(pca, "pca") # логируем обученные pca и модель
        mlflow.sklearn.log_model(model, "model")