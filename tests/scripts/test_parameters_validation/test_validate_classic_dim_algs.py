import pytest
import numpy as np
from scripts.parameters_validation.validate_classic_dim_algs import validate_dimensionality_config, validate_loaded_arrays

class TestValidateDimensionalityConfig:
    """Класс для тестирования метода validate_dimensionality_config"""

    def test_none_dim_alg_type(self):
        """Тест: None значение обязательного параметра dimensionally_alg_type"""
        with pytest.raises(ValueError, match="dimensionally_alg_type не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type=None,
                dim_arg_hyperparams={},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_empty_dim_alg_type(self):
        """Тест: пустое строковое значение параметра dimensionally_alg_type"""
        with pytest.raises(ValueError, match="dimensionally_alg_type не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type="  ",
                dim_arg_hyperparams={},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_incorrect_dim_alg_type_value(self):
        """Тест: недопустимое значение dimensionally_alg_type"""
        with pytest.raises(ValueError, match=f"Неизвестный алгоритм: incorrect value. "):
            validate_dimensionality_config(
                dimensionally_alg_type="incorrect value",
                dim_arg_hyperparams={},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_none_dim_arg_hyperparams(self):
        """Тест: None значение параметра dim_arg_hyperparams"""
        with pytest.raises(ValueError, match="dim_arg_hyperparams должен быть непустым словарем"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams=None,
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_dim_arg_hyperparams_incorrect_type(self):
        """Тест: недопустимы тип параметра dim_arg_hyperparams"""
        with pytest.raises(ValueError, match="dim_arg_hyperparams должен быть непустым словарем"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams=12,
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_dim_arg_hyperparams_empty_dict(self):
        """Тест: пустой словарь вместо параметра dim_arg_hyperparams"""
        with pytest.raises(ValueError, match="dim_arg_hyperparams должен быть непустым словарем"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_empty_bucket_name(self):
        """Тест: пустое название бакета"""
        with pytest.raises(ValueError, match=f"bucket_name не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"pca_components": 120},
                bucket_name="   ",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_none_bucket_name(self):
        """Тест: None значение вместо названия бакета"""
        with pytest.raises(ValueError, match=f"bucket_name не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"pca_components": 120},
                bucket_name=None,
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_empty_processed_prefix(self):
        """Тест: пустой префикс директории"""
        with pytest.raises(ValueError, match=f"processed_prefix не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"pca_components": 120},
                bucket_name="mri",
                processed_prefix="   ",
                local_data_dir="test_dir"
            )

    def test_none_processed_prefix(self):
        """Тест: None значение для параметра processed_prefix"""
        with pytest.raises(ValueError, match=f"processed_prefix не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"pca_components": 120},
                bucket_name="mri",
                processed_prefix=None,
                local_data_dir="test_dir"
            )

    def test_empty_local_data_dir(self):
        """Тест: пустое значение параметра local_data_dir"""
        with pytest.raises(ValueError, match=f"local_data_dir не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"pca_components": 120},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="   "
            )

    def test_none_local_data_dir(self):
        """Тест: None значение параметра local_data_dir"""
        with pytest.raises(ValueError, match=f"local_data_dir не может быть пустым"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"pca_components": 120},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir=None
            )

    def test_missing_pca_components_with_pca_alg(self):
        """Тест: пропущен обязательный гиперпараметр pca_components для работы pca"""
        with pytest.raises(ValueError, match="Для PCA требуется 'pca_components'"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"another_parameter": 120},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_incorrect_type_of_pca_components(self):
        """Тест: неверный тип гиперпараметра pca_components"""
        with pytest.raises(ValueError, match="pca_components должен быть числом"):
            validate_dimensionality_config(
                dimensionally_alg_type="pca",
                dim_arg_hyperparams={"pca_components": "123"},
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_missing_required_tsne_params(self):
        """Тест: пропущены обязательные параметры для работы t-SNE"""
        dim_arg_hyperparams = {
            "n_components": 120,
            "perplexity": 12,
            "early_exaggeration": True,
        }
        required = {"n_components", "perplexity", "early_exaggeration", "learning_rate"}
        missing = required - dim_arg_hyperparams.keys()
        with pytest.raises(ValueError, match=f"Для t-SNE отсутствуют параметры: learning_rate"):
            validate_dimensionality_config(
                dimensionally_alg_type="tsne",
                dim_arg_hyperparams=dim_arg_hyperparams,
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_missing_required_umap_params(self):
        """Тест: пропущены обязательные параметры для работы umap"""
        dim_arg_hyperparams = {
            "n_components": 120,
        }
        required = {"n_neighbors", "min_dist", "n_components", "metric", "spread"}
        missing = required - dim_arg_hyperparams.keys()
        with pytest.raises(ValueError, match=f"Для UMAP отсутствуют параметры: {", ".join(sorted(missing))}"):
            validate_dimensionality_config(
                dimensionally_alg_type="umap",
                dim_arg_hyperparams=dim_arg_hyperparams,
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_correct_params(self):
        """Тест: все параметры метода прошли проверки -> None"""
        dim_arg_hyperparams = {
            "pca_components": 2,
        }

        validate_dimensionality_config(
            dimensionally_alg_type="pca",
            dim_arg_hyperparams=dim_arg_hyperparams,
            bucket_name="mri",
            processed_prefix="processed",
            local_data_dir="test_dir",
        )

class TestValidateLoadedArrays:
    """Класс для проверки метода validate_loaded_arrays"""
    def test_empty_x_size(self):
        """Тест: пустой массив признаков"""
        X = np.array([])
        y = np.ones(10)
        with pytest.raises(ValueError, match="Загружен пустой массив X"):
            validate_loaded_arrays(X, y)

    def test_empty_y_size(self):
        """Тест: пустой массив target переменных"""
        X = np.random.randint(0, 1, (30, 30))
        y = np.array([])
        with pytest.raises(ValueError, match="Загружен пустой массив y"):
            validate_loaded_arrays(X, y)

    def test_mismatch_of_dimensions(self):
        """Тест: несоответствие размеров полученных массивов"""
        X = np.random.randint(0, 1, (30, 30))
        y = np.ones(10)
        with pytest.raises(ValueError,
                           match=rf"Несоответствие размерностей: X\[{X.shape[0]}\], y\[{y.shape[0]}\]"):
            validate_loaded_arrays(X, y)

    def test_correct_arrays(self):
        """Тест: массивы прошли проверку"""
        X = np.random.randint(0, 1, (30, 30))
        y = np.ones(30)
        validate_loaded_arrays(X, y)