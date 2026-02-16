from typing import Dict
import numpy as np

def validate_dimensionality_config(
    dimensionally_alg_type: str,
    dim_arg_hyperparams: Dict,
    bucket_name: str,
    processed_prefix: str,
    local_data_dir: str,
) -> None:
    """
        Валидирует конфигурацию алгоритма снижения размерности и связанные параметры.

        Поддерживаемые алгоритмы:
        - pca
        - tsne
        - umap
        - TDA (В РАЗРАБОТКЕ)

        Args:
            dimensionally_alg_type (str):
                Тип алгоритма снижения размерности.
            dim_arg_hyperparams (Dict):
                Словарь гиперпараметров выбранного алгоритма.
            bucket_name (str):
                Имя бакета для хранения данных.
            processed_prefix (str):
                Префикс директории для обработанных данных.
            local_data_dir (str):
                Локальная директория для хранения данных.

        Returns: None
    """
    valid_algorithms = {"pca", "tsne", "umap", "TDA"}

    if not dimensionally_alg_type or not dimensionally_alg_type.strip():
        raise ValueError("dimensionally_alg_type не может быть пустым")

    if dimensionally_alg_type not in valid_algorithms:
        raise ValueError(
            f"Неизвестный алгоритм: {dimensionally_alg_type}. "
            f"Допустимые: {', '.join(valid_algorithms)}"
        )

    if not isinstance(dim_arg_hyperparams, dict) or not dim_arg_hyperparams:
        raise ValueError("dim_arg_hyperparams должен быть непустым словарем")

    if not bucket_name or not bucket_name.strip():
        raise ValueError("bucket_name не может быть пустым")

    if not processed_prefix or not processed_prefix.strip():
        raise ValueError("processed_prefix не может быть пустым")

    if not local_data_dir or not local_data_dir.strip():
        raise ValueError("local_data_dir не может быть пустым")

    if dimensionally_alg_type == "pca":
        if "pca_components" not in dim_arg_hyperparams:
            raise ValueError("Для PCA требуется 'pca_components'")
        if not isinstance(dim_arg_hyperparams["pca_components"], (int, float)):
            raise ValueError("pca_components должен быть числом")

    elif dimensionally_alg_type == "tsne":
        required = {"n_components", "perplexity", "early_exaggeration", "learning_rate"}
        missing = required - dim_arg_hyperparams.keys()
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Для t-SNE отсутствуют параметры: {missing_str}"
            )

    elif dimensionally_alg_type == "umap":
        required = {"n_neighbors", "min_dist", "n_components", "metric", "spread"}
        missing = required - dim_arg_hyperparams.keys()
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Для UMAP отсутствуют параметры: {missing_str}"
            )

    return None

def validate_loaded_arrays(X: np.ndarray,
                           y: np.ndarray) -> None:
    """
        Проверяет корректность загруженных массивов признаков и меток.

        Args:
            X (np.ndarray):
                Массив признаков (shape: [n_samples, n_features]).
            y (np.ndarray):
                Массив целевых меток (shape: [n_samples]).

        Returns: None
    """
    if X.size == 0:
        raise ValueError("Загружен пустой массив X")

    if y.size == 0:
        raise ValueError("Загружен пустой массив y")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Несоответствие размерностей: X[{X.shape[0]}], y[{y.shape[0]}]"
        )

    return None