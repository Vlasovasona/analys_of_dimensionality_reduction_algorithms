import os
from typing import Tuple, List
import numpy as np
from numpy import ndarray
import mlflow
import logging
import mlflow.sklearn
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from scripts.parameters_validation.validate_classic_dim_algs import validate_dimensionality_config, validate_loaded_arrays

logger = logging.getLogger(__name__)

def _prepare_train_test_datasets(
    bucket_name: str,
    processed_prefix: str,
    local_data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Загружает все батчи из S3, делает train/test split
    и сохраняет итоговые датасеты обратно в S3
    """

    s3 = S3Hook(aws_conn_id="s3")
    base_tmp = Path(f"/tmp/{local_data_dir}")
    base_tmp.mkdir(parents=True, exist_ok=True)

    keys = s3.list_keys(bucket_name, f"{processed_prefix}/processed/")
    if not keys:
        raise RuntimeError("Нет данных для split")

    X_list, y_list = [], []

    for key in keys:
        if key.endswith(".npy") and "X_" in key:
            local_x = base_tmp / os.path.basename(key)
            local_y = base_tmp / os.path.basename(key.replace("X_", "y_"))

            s3.get_conn().download_file(bucket_name, key, str(local_x))
            s3.get_conn().download_file(bucket_name, key.replace("X_", "y_"), str(local_y))

            X_list.append(np.load(local_x))
            y_list.append(np.load(local_y))

    X = np.concatenate(X_list).reshape(len(np.concatenate(X_list)), -1)
    y = np.concatenate(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # сохраняем локально
    np.save(base_tmp / "X_train.npy", X_train)
    np.save(base_tmp / "X_test.npy", X_test)
    np.save(base_tmp / "y_train.npy", y_train)
    np.save(base_tmp / "y_test.npy", y_test)

    # загружаем обратно в S3
    for name in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]:
        s3.load_file(
            filename=str(base_tmp / name),
            key=f"{processed_prefix}/final/{name}",
            bucket_name=bucket_name,
            replace=True,
        )

def load_train_test_from_s3(
    bucket_name: str,
    processed_prefix: str,
    local_data_dir: str,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Загружает train/test датасеты из S3

    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        s3 = S3Hook(aws_conn_id="s3")
        logger.info("Подключение к S3 успешно")
    except NoCredentialsError:
        raise ConnectionError("Отсутствуют AWS credentials") from None
    except EndpointConnectionError:
        raise ConnectionError("Нет подключения к AWS endpoint")

    base_tmp = Path(f"/tmp/{local_data_dir}")
    base_tmp.mkdir(parents=True, exist_ok=True)

    files = {
        "X_train.npy": None,
        "X_test.npy": None,
        "y_train.npy": None,
        "y_test.npy": None,
    }

    for name in files.keys():
        s3_key = f"{processed_prefix}/final/{name}"
        local_path = base_tmp / name

        try:
            s3.get_conn().download_file(
                Bucket=bucket_name,
                Key=s3_key,
                Filename=str(local_path),
            )
            files[name] = np.load(local_path)
            logger.info(f"Загружен {s3_key}")
        except ClientError as e:
            raise FileNotFoundError(f"Не найден файл {s3_key} в S3") from e

    X_train = files["X_train.npy"].reshape(len(files["X_train.npy"]), -1)
    X_test = files["X_test.npy"].reshape(len(files["X_test.npy"]), -1)
    y_train = files["y_train.npy"]
    y_test = files["y_test.npy"]

    validate_loaded_arrays(X_train, y_train)
    validate_loaded_arrays(X_test, y_test)

    return X_train, X_test, y_train, y_test


def _train_dim_model(
    dimensionally_alg_type: str,
    dim_arg_hyperparams: dict,
    bucket_name: str,
    processed_prefix: str,
    local_data_dir: str,
) -> None:
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
    from scripts.dag_config import MLFLOW_URI, MLFLOW_EXPERIMENT_NAME

    mlflow_experiment_name = MLFLOW_EXPERIMENT_NAME
    mlflow_uri = MLFLOW_URI

    logger.info(f"Start learning {dimensionally_alg_type} with hyperparams: {dim_arg_hyperparams}")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    X_train, X_test, y_train, y_test = load_train_test_from_s3(
        bucket_name=bucket_name,
        processed_prefix=processed_prefix,
        local_data_dir=local_data_dir,
    )

    validate_loaded_arrays(X_train, y_train)

    with mlflow.start_run(run_name=f"{dimensionally_alg_type}"):

        if dimensionally_alg_type == "pca":
            n_components = dim_arg_hyperparams["pca_components"]
            max_components = min(X_train.shape[1], X_train.shape[0])
            n_components = min(n_components, max_components)

            model = PCA(n_components=n_components)

            X_train_new = model.fit_transform(X_train)
            X_test_new = model.transform(X_test)

            mlflow.log_metric(
                "explained_variance_ratio",
                float(sum(model.explained_variance_ratio_)),
            )

        elif dimensionally_alg_type == "umap":
            model = UMAP(**dim_arg_hyperparams)

            X_train_new = model.fit_transform(X_train)
            X_test_new = model.transform(X_test)


        elif dimensionally_alg_type == "TDA":  # реализовать!
            mlflow.log_param("TDA", "TDA")

        if model is not None:
            mlflow.sklearn.log_model(model, f"dimensionally_alg_type")

            for key, value in dim_arg_hyperparams.items():
                try:
                    mlflow.log_param(key, value)
                except Exception as e:
                    logger.warning(f"Не удалось залогировать параметр {key}: {e}")

        # Загружаем файл в S3
        s3 = S3Hook(aws_conn_id="s3")

        outputs = {
            f"X_{dimensionally_alg_type}_train.npy": X_train_new,
            f"X_{dimensionally_alg_type}_test.npy": X_test_new,
        }

        for name, array in outputs.items():
            if array is None:
                continue

            local_file = Path(name)
            np.save(local_file, array)

            s3.load_file(
                filename=str(local_file),
                key=f"{processed_prefix}/transformed/{name}",
                bucket_name=bucket_name,
                replace=True,
            )

            local_file.unlink()
