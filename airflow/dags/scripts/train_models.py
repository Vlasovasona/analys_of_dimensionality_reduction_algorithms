import os
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def load_data_from_s3(bucket_name="mri-dataset",
                      processed_prefix='mri',
                      local_data_dir="mri_train_data"):
    """
    Загружает батчи из S3 в память
    """
    s3 = S3Hook(aws_conn_id="s3")
    os.makedirs(f"/tmp/{local_data_dir}", exist_ok=True)

    keys = s3.list_keys(bucket_name,
                        f"{processed_prefix}/processed/") # получили список всех файлов внутри PROCESSED_PREFIX

    X_list, y_list = [], []

    for key in keys: # цикл по файлам в S3
        if key.endswith(".npy") and "X_" in key: # если это один из батчей X
            local_x = Path(f"/tmp/{local_data_dir}") / os.path.basename(key)
            local_y = Path(f"/tmp/{local_data_dir}") / os.path.basename(key.replace("X_", "y_"))

            # создаём директорию, если её нет
            local_x.parent.mkdir(parents=True, exist_ok=True)
            local_y.parent.mkdir(parents=True, exist_ok=True)

            # скачиваем батчи с данными для обучения и метками классов
            s3.get_conn().download_file(
                Bucket=bucket_name,
                Key=key,
                Filename=str(local_x)
            )
            s3.get_conn().download_file(
                Bucket=bucket_name,
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

def load_dim_model_from_s3():
    pass

def _train_model(
    model_type: str,
    dimensionally_alg_type: str,
    bucket_name: str = "mri-dataset",
    processed_prefix: str = 'mri',
    local_data_dir: str = "mri_train_data",
    mlflow_experiment_name: str = "mri-brain-tumor",
    mlflow_uri: str = "http://mlflow:5000",
):
    """
    Универсальная функция обучения модели для Airflow
    """
    mlflow.set_tracking_uri(mlflow_uri) # куда отправлять логи
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{dimensionally_alg_type}/{model_type}"):
        X, y = load_data_from_s3(bucket_name, processed_prefix, local_data_dir)

        mlflow.log_param("model_type", model_type) # логируем параметры обучения
        mlflow.log_param("original_dim", X.shape[1])

        alg_model =  load_dim_model_from_s3(bucket_name, processed_prefix, local_data_dir)
        X_new = alg_model.fit_transform(X)


        # создаем модели в зависимости от значения аргумента model_type
        if model_type == "logreg":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "svm":
            model = SVC(kernel="rbf")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_new, y)
        preds = model.predict(X_new)

        acc = accuracy_score(y, preds) # логируем метрики качества классификации
        conf_matrix = confusion_matrix(y, preds)
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("conf_matrix", conf_matrix)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(model, "model")