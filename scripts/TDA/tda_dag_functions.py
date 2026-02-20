from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import pickle
import numpy as np

from scripts.dag_config import BUCKET_NAME

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



def _vectorize_persistence_diagrams(test_size: float):
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

    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=42, shuffle=True
    )

    # Сохраняем локально
    train_path = os.path.join(TMP_DIR, "X_train.npy")
    test_path = os.path.join(TMP_DIR, "X_test.npy")
    np.save(train_path, X_train)
    np.save(test_path, X_test)

    # Загружаем на S3
    s3.load_file(
        filename=train_path,
        key=os.path.join(PROCESSED_PREFIX, "TDA_vectorized/X_train.npy"),
        bucket_name=BUCKET_NAME,
        replace=True
    )
    s3.load_file(
        filename=test_path,
        key=os.path.join(PROCESSED_PREFIX, "TDA_vectorized/X_test.npy"),
        bucket_name=BUCKET_NAME,
        replace=True
    )

    os.remove(train_path)
    os.remove(test_path)

    print(f"Готово! Train shape: {X_train.shape}, Test shape: {X_test.shape}")

