from scripts.TDA.create_persistence_diagram_functions import create_persistence_diagram, delete_noise_from_diag
from scripts.TDA.vectorize_diagram_functions import (features_lifetime, features_mid_lifetime,
    persistence_to_diagrams, betti_curves_from_persistence, triangle_function, persistence_landscape_1d)
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import io
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
            local_path=TMP_DIR,  # ← ТОЛЬКО ДИРЕКТОРИЯ
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
    pass
