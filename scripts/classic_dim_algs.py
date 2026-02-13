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
    mlflow_experiment_name: str = "default_name",
    mlflow_uri: str = "http://mlflow:5000",
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
        ValueError: При некорректных параметрах или отсутствии данных
        ConnectionError: При проблемах с подключением к AWS
        RuntimeError: При критических ошибках выполнения
    """
    logger.info(f"Start learning {dimensionally_alg_type} with hyperparams: {dim_arg_hyperparams}")

    try:
        if not dimensionally_alg_type:
            raise ValueError("dimensionally_alg_type не может быть пустым")

        valid_algorithms = ["pca", "tsne", "umap", "TDA"]
        if dimensionally_alg_type not in valid_algorithms:
            raise ValueError(f"Неизвестный тип алгоритма: {dimensionally_alg_type}. "
                             f"Допустимые значения: {valid_algorithms}")

        if not dim_arg_hyperparams:
            raise ValueError("dim_arg_hyperparams не может быть пустым")

        if not bucket_name:
            raise ValueError("bucket_name не может быть пустым")

        if not processed_prefix:
            raise ValueError("processed_prefix не может быть пустым")

        if not local_data_dir:
            raise ValueError("local_data_dir не может быть пустым")

        # Проверка обязательных параметров для каждого алгоритма
        if dimensionally_alg_type == "pca":
            if "pca_components" not in dim_arg_hyperparams:
                raise ValueError("Для PCA обязателен параметр 'pca_components'")
            if not isinstance(dim_arg_hyperparams["pca_components"], (int, float)):
                raise ValueError("pca_components должен быть числом")

        elif dimensionally_alg_type == "tsne":
            required_params = ["n_components", "perplexity", "early_exaggeration", "learning_rate"]
            missing = [p for p in required_params if p not in dim_arg_hyperparams]
            if missing:
                raise ValueError(f"Для t-SNE обязательны параметры: {missing}")

        elif dimensionally_alg_type == "umap":
            required_params = ["n_neighbors", "min_dist", "n_components", "metric", "spread"]
            missing = [p for p in required_params if p not in dim_arg_hyperparams]
            if missing:
                raise ValueError(f"Для UMAP обязательны параметры: {missing}")

    except ValueError as e:
        raise

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    X, y = load_data_from_s3(bucket_name, processed_prefix, local_data_dir)

    if X.size == 0:
        raise ValueError("Загружен пустой массив X")
    if y.size == 0:
        raise ValueError("Загружен пустой массив y")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Несоответствие размерностей: X[{X.shape[0]}], y[{y.shape[0]}]")

    with mlflow.start_run(run_name=f"{dimensionally_alg_type}"):
        if dimensionally_alg_type == "pca":
            n_components = dim_arg_hyperparams["pca_components"]

            max_components = min(X.shape[1], X.shape[0])
            if n_components > max_components:
                logger.warning(
                    f"pca_components={n_components} больше максимально возможного {max_components}. Уменьшено до {max_components}")
                n_components = max_components

            mlflow.log_param("pca_components", n_components)

            model = PCA(n_components=n_components)
            X_new = model.fit_transform(X)

            explained_variance = sum(model.explained_variance_ratio_)
            mlflow.log_metric("explained_variance_ratio", explained_variance)

        elif dimensionally_alg_type == "tsne":
            logger.info("Обучение t-SNE")
            max_samples = 10_000
            if X.shape[0] > max_samples:
                # слишком большой объем данных, t-SNE применять может быть нецелесообразно
                logger.warning(f"t-SNE с {X.shape[0]} сэмплами может быть медленным. Рекомендуется ≤ {max_samples}")

            try:
                model = TSNE(**dim_arg_hyperparams)
                X_new = model.fit_transform(X)
                logger.info(f"t-SNE выполнен: X_new.shape={X_new.shape}")
            except Exception as e:
                logger.error(f"Ошибка обучения t-SNE: {e}")
                raise RuntimeError(f"Ошибка обучения t-SNE: {e}") from e


        elif dimensionally_alg_type == "umap":
            logger.info("Обучение umap")

            try:
                model = UMAP(**dim_arg_hyperparams)
                X_new = model.fit_transform(X)
            except Exception as e:
                logger.error(f"Ошибка обучения UMAP: {e}")
                raise RuntimeError(f"Ошибка обучения UMAP: {e}") from e

        elif dimensionally_alg_type == "TDA":  # реализовать!
            mlflow.log_param("TDA", "TDA")

        else:
            ValueError(f"Unknown dimensionally algorithm type: {dimensionally_alg_type}")

        if model is not None:
            mlflow.sklearn.log_model(model, f"dimensionally_alg_type")

            for key, value in dim_arg_hyperparams.items():
                try:
                    mlflow.log_param(key, value)
                except Exception as e:
                    logger.warning(f"Не удалось залогировать параметр {key}: {e}")

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
