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


def load_data_from_s3(bucket_name,
                      processed_prefix,
                      local_data_dir):
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

def _load_and_concat_targets_from_s3(bucket_name,
                                      processed_prefix,
                                      local_data_dir):
    """
    Загружает батчи targets в память и соединяет для обработки алгоритмом
    """
    s3 = S3Hook(aws_conn_id="s3")
    os.makedirs(f"/tmp/{local_data_dir}", exist_ok=True)

    keys = s3.list_keys(bucket_name,
                        f"{processed_prefix}/processed/")  # получили список всех файлов внутри PROCESSED_PREFIX

    y_list = []

    for key in keys:  # цикл по файлам в S3
        if key.endswith(".npy") and "X_" in key:  # если это один из батчей X
            local_y = Path(f"/tmp/{local_data_dir}") / os.path.basename(key.replace("X_", "y_"))
            local_y.parent.mkdir(parents=True, exist_ok=True)
            s3.get_conn().download_file(
                Bucket=bucket_name,
                Key=key.replace("X_", "y_"),
                Filename=str(local_y)
            )
            y = np.load(local_y)
            y_list.append(y)
    y = np.concatenate(y_list)

    output_filename = f"y_transformed.npy"
    np.save(output_filename, y)

    # Загружаем файл в S3
    s3 = S3Hook(aws_conn_id="s3")
    s3.load_file(
        filename=output_filename,
        key=f"{processed_prefix}/transformed/{output_filename}",
        bucket_name=bucket_name,
        replace=True
    )

    print(f"Файл {output_filename} успешно загружен в S3.")
    os.remove(output_filename)


def _train_dim_model(
    dimensionally_alg_type: str,
    dim_arg_hyperparams: dict,
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

    with mlflow.start_run(run_name=f"{dimensionally_alg_type}"):
        X, y = load_data_from_s3(bucket_name, processed_prefix, local_data_dir)

        if dimensionally_alg_type == 'pca':
            mlflow.log_param("pca_components", dim_arg_hyperparams['pca_components'])

            pca = PCA(n_components=dim_arg_hyperparams['pca_components'])
            X_new = pca.fit_transform(X)


        elif dimensionally_alg_type == 'tsne':
            mlflow.log_param("n_components", dim_arg_hyperparams['n_components'])
            mlflow.log_param("perplexity", dim_arg_hyperparams['perplexity'])
            mlflow.log_param("early_exaggeration", dim_arg_hyperparams['early_exaggeration'])
            mlflow.log_param("learning_rate", dim_arg_hyperparams['learning_rate'])
            mlflow.log_param("metric", dim_arg_hyperparams['metric'])
            mlflow.log_param("init", dim_arg_hyperparams['init'])
            mlflow.log_param("angle", dim_arg_hyperparams['angle'])

            tsne = TSNE(
                n_components=dim_arg_hyperparams['n_components'],  # 2-компонентное представление
                perplexity=dim_arg_hyperparams['perplexity'],
                early_exaggeration=dim_arg_hyperparams['early_exaggeration'],  # Умножение дальних расстояний на раннем этапе
                learning_rate=dim_arg_hyperparams['learning_rate'],  # авто-выбор скорости обучения
                metric=dim_arg_hyperparams['metric'],  # евклидово расстояние
                init=dim_arg_hyperparams['init'],
                angle=dim_arg_hyperparams['angle']  # баланс между скоростью и точностью
            )
            X_new = tsne.fit_transform(X)

        elif dimensionally_alg_type == 'umap':
            mlflow.log_param("umap_components", dim_arg_hyperparams['n_components'])
            mlflow.log_param("min_dist", dim_arg_hyperparams['min_dist'])
            mlflow.log_param("n_neighbors", dim_arg_hyperparams['n_neighbors'])
            mlflow.log_param("metric", dim_arg_hyperparams['metric'])
            mlflow.log_param("spread", dim_arg_hyperparams['spread'])
            mlflow.log_param("low_memory", dim_arg_hyperparams['low_memory'])
            mlflow.log_param("init", dim_arg_hyperparams['init'])

            umap = UMAP(
                n_neighbors=dim_arg_hyperparams['n_neighbors'],  # количество соседей
                min_dist=dim_arg_hyperparams['min_dist'],  # плотные группировки близких элементов
                n_components=dim_arg_hyperparams['n_components'],  # двухкомпонентное представление
                metric=dim_arg_hyperparams['metric'],  #  евклидова метрика
                spread=dim_arg_hyperparams['spread'],  # нормальное распространение
                low_memory=dim_arg_hyperparams['low_memory'],  # экономия памяти при большой выборке
                init=dim_arg_hyperparams['init'],
            )
            X_new = umap.fit_transform(X)


        elif dimensionally_alg_type == 'TDA': # реализовать!
            mlflow.log_param("TDA", "TDA")


        else:
            ValueError(f"Unknown dimensionally algorithm type: {dimensionally_alg_type}")

        if dimensionally_alg_type == 'pca':
            mlflow.sklearn.log_model(pca, "pca")
        elif dimensionally_alg_type == 'tsne':
            mlflow.sklearn.log_model(tsne, "tsne")
        elif dimensionally_alg_type == 'umap':
            mlflow.sklearn.log_model(umap, "umap")

        output_filename = f"X_{dimensionally_alg_type}_transformed.npy"
        np.save(output_filename, X_new)

        # Загружаем файл в S3
        s3 = S3Hook(aws_conn_id="s3")
        s3.load_file(
            filename=output_filename,
            key=f"{processed_prefix}/transformed/{output_filename}",
            bucket_name=bucket_name,
            replace=True
        )

        print(f"Файл {output_filename} успешно загружен в S3.")

        os.remove(output_filename)