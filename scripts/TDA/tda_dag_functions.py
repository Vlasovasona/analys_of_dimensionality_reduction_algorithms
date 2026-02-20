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



def _vectorize_persistence_diagrams():
    """
    Векторизация диаграмм персистентвности для каждого канала
    Объединение трех каналов в один
    Загрузка в S3 в формате .npy
    """
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

    for key in keys:
        filename = os.path.basename(key)
        local_pd_path = os.path.join(TMP_DIR, filename)

        # Скачиваем батч PD
        tmp_path = s3.download_file(
            key=key,
            bucket_name=BUCKET_NAME,
            local_path=TMP_DIR,  # только директория
        )
        os.replace(tmp_path, local_pd_path)

        # Загружаем диаграммы
        with open(local_pd_path, "rb") as f:
            batch_pd_dicts = pickle.load(f)  # каждый элемент = {"gray":..., "sobel":..., "gaussian":...}

        batch_vectors = []
        for pd_dict in batch_pd_dicts:
            vecs_per_image = []
            for channel in ["gray", "sobel", "gaussian"]:
                channel_vec = vectorize_diagram(pd_dict[channel])
                vecs_per_image.append(channel_vec)
            batch_vectors.append(np.concatenate(vecs_per_image))

        X_vector = np.array(batch_vectors, dtype=np.float32)

        vec_filename = filename.replace("PD_", "VEC_").replace(".pkl", ".npy")
        local_vec_path = os.path.join(TMP_DIR, vec_filename)
        np.save(local_vec_path, X_vector)

        s3.load_file(
            filename=local_vec_path,
            key=f"{PROCESSED_PREFIX}TDA_vectorized/{vec_filename}",
            bucket_name=BUCKET_NAME,
            replace=True
        )

        # Чистим tmp
        os.remove(local_pd_path)
        os.remove(local_vec_path)

