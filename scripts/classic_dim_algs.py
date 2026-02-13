import os
from typing import Tuple, List
import numpy as np
from numpy import ndarray
import mlflow
import logging
import mlflow.sklearn
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

logger = logging.getLogger(__name__)

def load_data_from_s3(bucket_name: str,
                      processed_prefix: str,
                      local_data_dir: str) -> Tuple[ndarray, ndarray]:
    """
    Загружает батчи из S3 в память

    Args:
        bucket_name: Название S3 бакета
        processed_prefix: Префикс пути к обработанным данным в бакете
        local_data_dir: Имя локальной директории для временных файлов (создается в /tmp/)

    Returns:
        Tuple[ndarray, ndarray]: Кортеж из двух массивов numpy:
            - X: Массив признаков размерности (n_samples, n_features)
            - y: Массив меток классов размерности (n_samples,)

    Raises:
        ValueError: Если не найдены файлы с данными или бакет не существует
        botocore.exceptions.NoCredentialsError: Если нет доступа к AWS
        botocore.exceptions.ClientError: При ошибках S3 (бакет не найден, нет прав)
        FileNotFoundError: Если не удалось скачать файлы
    """
    if not bucket_name:
        raise ValueError("bucket_name не может быть пустым")
    if not processed_prefix:
        raise ValueError("processed_prefix не может быть пустым")
    if not local_data_dir:
        raise ValueError("local_data_dir не может быть пустым")

    try:
        s3 = S3Hook(aws_conn_id="s3")
        logger.info("Подключение к S3 успешно")
    except NoCredentialsError:
        raise ConnectionError("Отсутствуют учетные данные AWS") from None
    except EndpointConnectionError:
        raise ConnectionError("Нет подключения к AWS эндпоинту")
    except Exception as e:
        raise ConnectionError(f"Ошибка подключения к S3: {e}") from e

    os.makedirs(f"/tmp/{local_data_dir}", exist_ok=True)

    try:
        keys = s3.list_keys(
            bucket_name, f"{processed_prefix}/processed/"
        )  # получили список всех файлов внутри PROCESSED_PREFIX
        logger.info(f"Найдено {len(keys)} файлов в s3")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchBucket':
            raise ValueError(f"Бакет {bucket_name} не найден") from e
        elif error_code == 'AccessDenied':
            raise PermissionError(f"Нет доступа к бакету {bucket_name}") from e
        else:
            raise ConnectionError(f"Ошибка S3: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Ошибка получения списка файлов: {e}") from e

    X_list, y_list = [], []

    for key in keys:  # цикл по файлам в S3
        if key.endswith(".npy") and "X_" in key:  # если это один из батчей X
            local_x = Path(f"/tmp/{local_data_dir}") / os.path.basename(key)
            local_y = Path(f"/tmp/{local_data_dir}") / os.path.basename(key.replace("X_", "y_"))

            # создаём директорию, если её нет
            local_x.parent.mkdir(parents=True, exist_ok=True)
            local_y.parent.mkdir(parents=True, exist_ok=True)

            # скачиваем батчи с данными для обучения и метками классов
            s3.get_conn().download_file(Bucket=bucket_name, Key=key, Filename=str(local_x))
            s3.get_conn().download_file(Bucket=bucket_name, Key=key.replace("X_", "y_"), Filename=str(local_y))

            X = np.load(local_x)
            y = np.load(local_y)

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list)  # склеиваем все батчи
    y = np.concatenate(y_list)

    return X.reshape(len(X), -1), y  # разворачиваем картинки в векторы


def _load_and_concat_targets_from_s3(bucket_name: str,
                                    processed_prefix: str,
                                    local_data_dir: str):
    """
    Загружает батчи targets в память и соединяет для обработки алгоритмом

    Args:
        bucket_name: Название S3 бакета
        processed_prefix: Префикс пути к обработанным данным в бакете
        local_data_dir: Имя локальной директории для временных файлов (создается в /tmp/)

    Returns: None

    Raises:
        ValueError: Если не найдены файлы с данными или бакет не существует
        botocore.exceptions.NoCredentialsError: Если нет доступа к AWS
        botocore.exceptions.ClientError: При ошибках S3 (бакет не найден, нет прав)
        FileNotFoundError: Если не удалось скачать файлы
    """
    if not bucket_name:
        raise ValueError("bucket_name не может быть пустым")
    if not processed_prefix:
        raise ValueError("processed_prefix не может быть пустым")
    if not local_data_dir:
        raise ValueError("local_data_dir не может быть пустым")

    try:
        s3 = S3Hook(aws_conn_id="s3")
        logger.info("Подключение к S3 успешно")
    except NoCredentialsError:
        raise ConnectionError("Отсутствуют учетные данные AWS") from None
    except EndpointConnectionError:
        raise ConnectionError("Нет подключения к AWS эндпоинту")
    except Exception as e:
        raise ConnectionError(f"Ошибка подключения к S3: {e}") from e


    os.makedirs(f"/tmp/{local_data_dir}", exist_ok=True)

    try:
        keys = s3.list_keys(
            bucket_name, f"{processed_prefix}/processed/"
        )  # получили список всех файлов внутри PROCESSED_PREFIX
        logger.info(f"Найдено {len(keys)} файлов в s3")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchBucket':
            raise ValueError(f"Бакет {bucket_name} не найден") from e
        elif error_code == 'AccessDenied':
            raise PermissionError(f"Нет доступа к бакету {bucket_name}") from e
        else:
            raise ConnectionError(f"Ошибка S3: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Ошибка получения списка файлов: {e}") from e

    y_list = []

    for key in keys:  # цикл по файлам в S3
        if key.endswith(".npy") and "X_" in key:  # если это один из батчей X
            local_y = Path(f"/tmp/{local_data_dir}") / os.path.basename(key.replace("X_", "y_"))
            local_y.parent.mkdir(parents=True, exist_ok=True)
            s3.get_conn().download_file(Bucket=bucket_name, Key=key.replace("X_", "y_"), Filename=str(local_y))
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
        replace=True,
    )

    print(f"Файл {output_filename} успешно загружен в S3.")
    os.remove(output_filename)


def _train_dim_model(
    dimensionally_alg_type: str,
    dim_arg_hyperparams: dict,
    bucket_name: str,
    processed_prefix: str,
    local_data_dir: str,
    mlflow_experiment_name: str,
    mlflow_uri: str,
):
    """
    Универсальная функция обучения классического алгоритма понижения размерности

    Args:
        dimensionally_alg_type: Тип алгоритма понижения размерности,
        dim_arg_hyperparams: Словарь гиперпараметров для алгоритма
        bucket_name: Название S3 бакета
        processed_prefix: Префикс пути к обработанным данным в бакете
        local_data_dir: Имя локальной директории для временных файлов (создается в /tmp/)
        mlflow_experiment_name: Название эксперимента для MLflow
        mlflow_uri: URI для MLflow

    Returns: None

    Raises:
        ValueError: Если не найдены файлы с данными или бакет не существует
        botocore.exceptions.NoCredentialsError: Если нет доступа к AWS
        botocore.exceptions.ClientError: При ошибках S3 (бакет не найден, нет прав)
        FileNotFoundError: Если не удалось скачать файлы
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{dimensionally_alg_type}"):
        X, y = load_data_from_s3(bucket_name, processed_prefix, local_data_dir)

        if dimensionally_alg_type == "pca":
            mlflow.log_param("pca_components", dim_arg_hyperparams["pca_components"])

            pca = PCA(n_components=dim_arg_hyperparams["pca_components"])
            X_new = pca.fit_transform(X)

        elif dimensionally_alg_type == "tsne":
            mlflow.log_param("n_components", dim_arg_hyperparams["n_components"])
            mlflow.log_param("perplexity", dim_arg_hyperparams["perplexity"])
            mlflow.log_param("early_exaggeration", dim_arg_hyperparams["early_exaggeration"])
            mlflow.log_param("learning_rate", dim_arg_hyperparams["learning_rate"])
            mlflow.log_param("metric", dim_arg_hyperparams["metric"])
            mlflow.log_param("init", dim_arg_hyperparams["init"])
            mlflow.log_param("angle", dim_arg_hyperparams["angle"])

            tsne = TSNE(
                n_components=dim_arg_hyperparams["n_components"],  # 2-компонентное представление
                perplexity=dim_arg_hyperparams["perplexity"],
                early_exaggeration=dim_arg_hyperparams[
                    "early_exaggeration"
                ],  # Умножение дальних расстояний на раннем этапе
                learning_rate=dim_arg_hyperparams["learning_rate"],  # авто-выбор скорости обучения
                metric=dim_arg_hyperparams["metric"],  # евклидово расстояние
                init=dim_arg_hyperparams["init"],
                angle=dim_arg_hyperparams["angle"],  # баланс между скоростью и точностью
            )
            X_new = tsne.fit_transform(X)

        elif dimensionally_alg_type == "umap":
            mlflow.log_param("umap_components", dim_arg_hyperparams["n_components"])
            mlflow.log_param("min_dist", dim_arg_hyperparams["min_dist"])
            mlflow.log_param("n_neighbors", dim_arg_hyperparams["n_neighbors"])
            mlflow.log_param("metric", dim_arg_hyperparams["metric"])
            mlflow.log_param("spread", dim_arg_hyperparams["spread"])
            mlflow.log_param("low_memory", dim_arg_hyperparams["low_memory"])
            mlflow.log_param("init", dim_arg_hyperparams["init"])

            umap = UMAP(
                n_neighbors=dim_arg_hyperparams["n_neighbors"],  # количество соседей
                min_dist=dim_arg_hyperparams["min_dist"],  # плотные группировки близких элементов
                n_components=dim_arg_hyperparams["n_components"],  # двухкомпонентное представление
                metric=dim_arg_hyperparams["metric"],  #  евклидова метрика
                spread=dim_arg_hyperparams["spread"],  # нормальное распространение
                low_memory=dim_arg_hyperparams["low_memory"],  # экономия памяти при большой выборке
                init=dim_arg_hyperparams["init"],
            )
            X_new = umap.fit_transform(X)

        elif dimensionally_alg_type == "TDA":  # реализовать!
            mlflow.log_param("TDA", "TDA")

        else:
            ValueError(f"Unknown dimensionally algorithm type: {dimensionally_alg_type}")

        if dimensionally_alg_type == "pca":
            mlflow.sklearn.log_model(pca, "pca")
        elif dimensionally_alg_type == "tsne":
            mlflow.sklearn.log_model(tsne, "tsne")
        elif dimensionally_alg_type == "umap":
            mlflow.sklearn.log_model(umap, "umap")

        output_filename = f"X_{dimensionally_alg_type}_transformed.npy"
        np.save(output_filename, X_new)

        # Загружаем файл в S3
        s3 = S3Hook(aws_conn_id="s3")
        s3.load_file(
            filename=output_filename,
            key=f"{processed_prefix}/transformed/{output_filename}",
            bucket_name=bucket_name,
            replace=True,
        )

        print(f"Файл {output_filename} успешно загружен в S3.")

        os.remove(output_filename)
