from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

from scripts.parameters_validation.validate_classic_dim_algs import validate_loaded_arrays

from scripts.dag_config import BUCKET_NAME

logger = logging.getLogger(__name__)

TMP_DIR = "/tmp/mri_batches"
PROCESSED_PREFIX = "mri/processed/"

def _compute_persistence_diagrams() -> None:
    """
    Загружает TDA-батчи из S3,
    считает диаграммы персистентности,
    сохраняет результат в S3 (.pkl).
    """
    import os
    import pickle
    import numpy as np
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook
    from scripts.TDA.create_persistence_diagram_functions import compute_pd_for_multichannel_image

    s3 = S3Hook(aws_conn_id="s3")

    # префикс на S3, где лежат батчи TDA
    tda_prefix = os.path.join(PROCESSED_PREFIX, "TDA")  # -> "mri/processed/TDA"
    os.makedirs(TMP_DIR, exist_ok=True)

    keys = sorted(
        k for k in s3.list_keys(BUCKET_NAME, prefix=tda_prefix)
        if os.path.basename(k).startswith("X_") and k.endswith(".npy")
    )

    for key in keys:
        filename = os.path.basename(key)
        final_path = os.path.join(TMP_DIR, filename)

        tmp_path = s3.download_file(
            key=key,
            bucket_name=BUCKET_NAME,
            local_path=TMP_DIR,
        )

        os.replace(tmp_path, final_path)

        X = np.load(final_path)

        batch_pd = [
            compute_pd_for_multichannel_image(x)
            for x in X
        ]

        pd_path = final_path.replace("X_", "PD_").replace(".npy", ".pkl")

        with open(pd_path, "wb") as f:
            pickle.dump(batch_pd, f)

        s3.load_file(
            filename=pd_path,
            key=f"{PROCESSED_PREFIX}TDA_PD/{os.path.basename(pd_path)}",
            bucket_name=BUCKET_NAME,
            replace=True
        )

        os.remove(final_path)
        os.remove(pd_path)


def _vectorize_persistence_diagrams(test_size: float, **context):
    """
    Векторизация диаграмм персистентвности для каждого канала
    Объединение трех каналов в один
    Загрузка в S3 в формате .npy
    """
    from sklearn.model_selection import train_test_split
    from scripts.TDA.vectorize_diagram_functions import vectorize_diagram, persistence_to_diagrams

    s3 = S3Hook(aws_conn_id="s3")
    os.makedirs(TMP_DIR, exist_ok=True)

    # Префикс на S3, где лежат батчи PD
    pd_prefix = os.path.join(PROCESSED_PREFIX, "TDA_PD")
    keys = sorted(
        k for k in s3.list_keys(BUCKET_NAME, prefix=pd_prefix)
        if k.endswith(".pkl")
    )

    if not keys:
        raise Exception(f"Нет диаграмм в бакете {pd_prefix}")

    all_vectors = []

    for key in keys:
        filename = os.path.basename(key)
        local_pd_path = os.path.join(TMP_DIR, filename)

        # Скачиваем батч PD
        tmp_path = s3.download_file(
            key=key,
            bucket_name=BUCKET_NAME,
            local_path=TMP_DIR,
        )
        os.replace(tmp_path, local_pd_path)

        # Загружаем диаграммы
        with open(local_pd_path, "rb") as f:
            batch_pd_dicts = pickle.load(f)  # каждый элемент = {"gray":..., "sobel":..., "gaussian":...}

        for pd_dict in batch_pd_dicts:
            vecs_per_image = []
            for channel in ["gray", "sobel", "gaussian"]:
                channel_vec = vectorize_diagram(pd_dict[channel])
                vecs_per_image.append(channel_vec)
            all_vectors.append(np.concatenate(vecs_per_image))

        os.remove(local_pd_path)

    X = np.array(all_vectors, dtype=np.float32)

    # Сохраняем локально
    X_path = os.path.join(TMP_DIR, "X.npy")
    np.save(X_path, X)

    # Загружаем на S3
    s3.load_file(
        filename=X_path,
        key=os.path.join(PROCESSED_PREFIX, "TDA_vectorized/X.npy"),
        bucket_name=BUCKET_NAME,
        replace=True
    )

    os.remove(X_path)

    print(f"Готово! X shape: {X.shape}")


def load_TDA_data_from_s3(bucket_name="mri-dataset", processed_prefix="mri", local_data_dir="mri_train_data"):
    """
    Загружает train/test датасеты из S3

    Returns:
        X_train, X_test, y_train, y_test
    """
    s3 = S3Hook(aws_conn_id="s3")
    base_tmp = Path(f"/tmp/{local_data_dir}")
    base_tmp.mkdir(parents=True, exist_ok=True)

    files = {
        "X_train.npy": None,
        "X_test.npy": None,
        "y_train.npy": None,
        "y_test.npy": None,
    }

    for name in files.keys():
        s3_key = f"{processed_prefix}/final/TDA/{name}"
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


def _train_TDA_models(
        model_type: str,
        bucket_name: str = "mri-dataset",
        processed_prefix: str = "mri",
        local_data_dir: str = "mri_train_data",
        mlflow_experiment_name: str = "mri-brain-tumor",
        mlflow_uri: str = "http://mlflow:5000",
):
    """
    Универсальная функция обучения модели classic ML + TDA
    """
    import mlflow
    import mlflow.sklearn

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=f"TDA/{model_type}"):

        X_train, X_test, y_train, y_test = load_TDA_data_from_s3(
            bucket_name, processed_prefix, local_data_dir
        )

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("original_dim", X_train.shape[1])
        mlflow.log_param("scaler", "StandardScaler")


        if model_type == "logreg":
            pipeline = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear"
                ))
            ])

            param_grid = {
                "clf__C": [1.0, 10.0],
                "clf__penalty": ["l1"],
            }

        elif model_type == "svm":
            pipeline = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    class_weight="balanced"
                ))
            ])

            param_grid = {
                "clf__C": [0.1, 1, 10, 100],
                "clf__gamma": ["scale", 0.1, 0.01, 0.001],
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")


        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(best_model, "model")

def _prepare_train_test_datasets_tda(
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

    keys = s3.list_keys(bucket_name, f"{processed_prefix}/processed/TDA")
    if not keys:
        raise RuntimeError("Нет данных для split")

    y_list = []

    for key in keys:
        if key.endswith(".npy") and "y_" in key:
            local_y = base_tmp / os.path.basename(key)

            s3.get_conn().download_file(bucket_name, key, str(local_y))
            y_list.append(np.load(local_y))

    y = np.concatenate(y_list)

    name = "X.npy"

    s3_key = f"{processed_prefix}/processed/TDA_vectorized/{name}"
    local_path = base_tmp / name

    s3.get_conn().download_file(
        Bucket=bucket_name,
        Key=s3_key,
        Filename=str(local_path),
    )
    X = np.load(local_path)
    X_final = X.reshape(len(X), -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
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
            key=f"{processed_prefix}/final/TDA/{name}",
            bucket_name=bucket_name,
            replace=True,
        )