import pytest
from scripts.parameters_validation.dag_config_validation import validate_storage_config

class TestValidateStorageConfig:
    def test_empty_bucket_name(self):
        """Тест: пустое название бакета"""
        with pytest.raises(ValueError, match=f"bucket_name не может быть пустым"):
            validate_storage_config(
                bucket_name="   ",
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_none_bucket_name(self):
        """Тест: None значение вместо названия бакета"""
        with pytest.raises(ValueError, match=f"bucket_name не может быть пустым"):
            validate_storage_config(
                bucket_name=None,
                processed_prefix="processed",
                local_data_dir="test_dir"
            )

    def test_empty_processed_prefix(self):
        """Тест: пустой префикс директории"""
        with pytest.raises(ValueError, match=f"processed_prefix не может быть пустым"):
            validate_storage_config(
                bucket_name="mri",
                processed_prefix="   ",
                local_data_dir="test_dir"
            )

    def test_none_processed_prefix(self):
        """Тест: None значение для параметра processed_prefix"""
        with pytest.raises(ValueError, match=f"processed_prefix не может быть пустым"):
            validate_storage_config(
                bucket_name="mri",
                processed_prefix=None,
                local_data_dir="test_dir"
            )

    def test_empty_local_data_dir(self):
        """Тест: пустое значение параметра local_data_dir"""
        with pytest.raises(ValueError, match=f"local_data_dir не может быть пустым"):
            validate_storage_config(
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir="   "
            )

    def test_none_local_data_dir(self):
        """Тест: None значение параметра local_data_dir"""
        with pytest.raises(ValueError, match=f"local_data_dir не может быть пустым"):
            validate_storage_config(
                bucket_name="mri",
                processed_prefix="processed",
                local_data_dir=None
            )

    def test_correct_parameters(self):
        """Тест: все параметры подключения переданы корректно"""
        validate_storage_config(
            bucket_name="mri",
            processed_prefix="processed",
            local_data_dir="test_dir"
        )
