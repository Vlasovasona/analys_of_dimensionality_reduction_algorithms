import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Tuple
from numpy import ndarray
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def load_dim_data_from_s3(
    bucket_name="mri-dataset", processed_prefix="mri", local_data_dir="mri_train_data", dim_alg_name="pca"
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Загружает данные, обработанные алгоритмами из бакета S3
    """
    s3 = S3Hook(aws_conn_id="s3")

    local_path = f"/tmp/{local_data_dir}"
    os.makedirs(local_path, exist_ok=True)

    s3_key_X_train = f"{processed_prefix}/transformed/X_{dim_alg_name}_train.npy"
    s3_key_X_test = f"{processed_prefix}/transformed/X_{dim_alg_name}_test.npy"
    s3_key_y_train = f"{processed_prefix}/final/y_train.npy"
    s3_key_y_test = f"{processed_prefix}/final/y_test.npy"

    local_file_X_train = os.path.join(local_path, f"X_{dim_alg_name}_train.npy")
    local_file_X_test = os.path.join(local_path, f"X_{dim_alg_name}_test.npy")

    local_file_y_train = os.path.join(local_path, f"y_{dim_alg_name}_train.npy")
    local_file_y_test = os.path.join(local_path, f"y_{dim_alg_name}_test.npy")

    try:
        s3.get_conn().download_file(Bucket=bucket_name, Key=s3_key_X_train, Filename=local_file_X_train)
        print(f"Файл {s3_key_X_train} успешно загружен в {local_file_X_train}")

        s3.get_conn().download_file(Bucket=bucket_name, Key=s3_key_X_test, Filename=local_file_X_test)
        print(f"Файл {s3_key_X_test} успешно загружен в {local_file_X_test}")

        X_train, X_test = np.load(local_file_X_train), np.load(local_file_X_test)

        try:
            s3.get_conn().download_file(Bucket=bucket_name, Key=s3_key_y_train, Filename=local_file_y_train)
            print(f"Файл {s3_key_y_train} успешно загружен в {local_file_y_train}")
            y_train = np.load(local_file_y_train)

            s3.get_conn().download_file(Bucket=bucket_name, Key=s3_key_y_test, Filename=local_file_y_test)
            print(f"Файл {s3_key_y_test} успешно загружен в {local_file_y_test}")
            y_test = np.load(local_file_y_test)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Файл с метками не найден: {e}")
            raise

    except Exception as e:
        print(f"Ошибка при загрузке файла {s3_key_X_train}: {e}")
        raise


def _train_model(
    dimensionally_alg_type: str,
    model_type: str,
    bucket_name: str = "mri-dataset",
    processed_prefix: str = "mri",
    local_data_dir: str = "mri_train_data",
    mlflow_experiment_name: str = "mri-brain-tumor",
    mlflow_uri: str = "http://mlflow:5000",
):
    """
    Универсальная функция обучения модели classic ML
    """
    mlflow.set_tracking_uri(mlflow_uri)  # куда отправлять логи
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{dimensionally_alg_type}/{model_type}"):
        X_train, X_test, y_train, y_test = load_dim_data_from_s3(bucket_name, processed_prefix, local_data_dir, dimensionally_alg_type)

        mlflow.log_param("model_type", model_type)  # логируем параметры обучения
        mlflow.log_param("original_dim", X_train.shape[1])

        # создаем модели в зависимости от значения аргумента model_type
        if model_type == "logreg":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "svm":
            model = SVC(kernel="rbf")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)  # логируем метрики качества классификации
        precision = precision_score(y_true=y_test, y_pred=preds, average="macro")
        recall = recall_score(y_true=y_test, y_pred=preds, average="macro")
        f1 = f1_score(y_true=y_test, y_pred=preds, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(model, "model")
