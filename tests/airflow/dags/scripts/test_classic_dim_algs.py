import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
import os
import tempfile
from scripts.classic_dim_algs import load_data_from_s3


class TestLoadDataFromS3:
    @patch("scripts.classic_dim_algs.S3Hook") # декораторы применяются снизу вверх
    @patch("scripts.classic_dim_algs.os.makedirs") # аргументы передаются сверху вниз
    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.Path.mkdir")
    def test_load_data_from_s3_success(self,
                                        mock_path_mkdir,
                                        mock_np_load,
                                        mock_makedirs,
                                        mock_s3_hook):
        """Тест успешной загрузки данных из S3"""

        # настройка моков
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        # мокаем список ключей
        mock_s3.list_keys.return_value = [
            "mri/processed/X_batch_1.npy",
            "mri/processed/X_batch_2.npy",
            "mri/processed/y_batch_1.npy",
        ]

        mock_s3_conn = MagicMock()
        mock_s3.get_conn.return_value = mock_s3_conn
        mock_s3_conn.download_file.return_value = None # заглушили метод download_file - ничего не делает

        # мокаем numpy.load для возврата тестовых данных
        X_mock = np.random.rand(10, 64, 64, 3).reshape(10, -1) # создаем 10 образов 64х64х3, преобразуем в плоский массив
        y_mock = np.random.randint(0, 2, 10) # 10 случайных targets 0 или 1

        # side_effect позволяет вернуть разные значения при последовательных вызовах
        mock_np_load.side_effect = [
            X_mock, y_mock,  # первый батч
            X_mock, y_mock  # второй батч
        ]

        # вызов тестируемой функции
        X, y = load_data_from_s3(
            bucket_name="test-bucket",
            processed_prefix="mri",
            local_data_dir="test_dir"
        )

        # проверки
        assert X is not None
        assert y is not None

        assert X.shape[0] == 20  # 2 батча по 10
        assert y.shape[0] == 20

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        # Проверяем, что list_keys вызван с правильными параметрами
        mock_s3.list_keys.assert_called_once_with(
            "test-bucket",
            "mri/processed/"
        )

        # проверяем, что download_file вызывался 4 раза (2 батча * (X + y))
        assert mock_s3_conn.download_file.call_count == 4

        # проверка, что создана директория
        mock_makedirs.assert_called_once()

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_empty_bucket_name(self, mock_s3_hook):
        """ Тест: пустое имя бакета -> ValueError """
        with pytest.raises(ValueError, match="bucket_name не может быть пустым"):
            load_data_from_s3(
                bucket_name="",
                processed_prefix="mri",
                local_data_dir="test_dir"
            )

        mock_s3_hook.assert_not_called()

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_empty_processed_prefix(self, mock_s3_hook):
        """ Тест: пустой processed prefix -> ValueError """
        with pytest.raises(ValueError, match="processed_prefix не может быть пустым"):
            load_data_from_s3(
                bucket_name="test-bucket",
                processed_prefix="",
                local_data_dir="test_dir"
            )

        mock_s3_hook.assert_not_called()

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_empty_local_data_dir(self, mock_s3_hook):
        """ Тест: пустой local_data_dir -> ValueError """
        with pytest.raises(ValueError, match="local_data_dir не может быть пустым"):
            load_data_from_s3(
                bucket_name="test-bucket",
                processed_prefix="mri",
                local_data_dir=""
            )

        mock_s3_hook.assert_not_called()

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_no_credentials_error(self, mock_s3_hook):
        """Тест: нет AWS credentials -> ConnectionError"""

        mock_s3_hook.side_effect = NoCredentialsError()

        with pytest.raises(ConnectionError, match="Отсутствуют учетные данные AWS"):
            load_data_from_s3(
                bucket_name="test-bucket",
                processed_prefix="mri",
                local_data_dir="test_dir"
            )

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_endpoint_connection_error(self, mock_s3_hook):
        """Тест: нет AWS подключения к эндпоинту -> EndpointConnectionError"""

        mock_s3_hook.side_effect = EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")

        with pytest.raises(ConnectionError, match="Нет подключения к AWS эндпоинту"):
            load_data_from_s3(
                bucket_name="test-bucket",
                processed_prefix="mri",
                local_data_dir="test_dir"
            )

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    def test_bucket_not_found(self, mock_makedirs, mock_s3_hook):
        """Тест: бакет не существует -> ValueError"""

        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        error_response = {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket does not exist'}}
        mock_s3.list_keys.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(ValueError, match="Бакет test-bucket не найден"):
            load_data_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    def test_access_denied(self, mock_makedirs, mock_s3_hook):
        """Тест: нет доступа к бакету -> PermissionError"""

        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3.list_keys.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(PermissionError, match="Нет доступа к бакету test-bucket"):
            load_data_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    def test_other_client_error(self, mock_makedirs, mock_s3_hook):
        """Тест: другая ошибка ClientError -> ConnectionError"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        error_response = {'Error': {'Code': 'InternalError', 'Message': 'Internal error'}}
        mock_s3.list_keys.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(ConnectionError, match="Ошибка S3"):
            load_data_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    def test_list_keys_generic_error(self, mock_makedirs, mock_s3_hook):
        """Тест: общая ошибка при list_keys -> RuntimeError"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        mock_s3.list_keys.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Ошибка получения списка файлов: Network error"):
            load_data_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    @patch("scripts.classic_dim_algs.Path.mkdir")
    def test_download_file_error(self, mock_path_mkdir, mock_makedirs, mock_s3_hook):
        """Тест: ошибка при скачивании файла"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        # Успешный list_keys
        mock_s3.list_keys.return_value = ["mri/processed/X_batch_1.npy"]

        # Ошибка при download_file
        mock_s3_conn = MagicMock()
        mock_s3.get_conn.return_value = mock_s3_conn
        mock_s3_conn.download_file.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchKey', 'Message': 'Key does not exist'}},
            'GetObject'
        )

        # должно упасть с ClientError
        with pytest.raises(ClientError):
            load_data_from_s3("test-bucket", "mri", "test_dir")


    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.Path.mkdir")
    def test_numpy_load_error(self, mock_path_mkdir, mock_np_load, mock_makedirs, mock_s3_hook):
        """Тест: ошибка загрузки numpy файла"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        mock_s3.list_keys.return_value = ["mri/processed/X_batch_1.npy"]

        mock_s3_conn = MagicMock()
        mock_s3.get_conn.return_value = mock_s3_conn
        mock_s3_conn.download_file.return_value = None

        # Ошибка при np.load
        mock_np_load.side_effect = Exception("Corrupted numpy file")

        with pytest.raises(Exception, match="Corrupted numpy file"):
            load_data_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.Path.mkdir")
    @patch("scripts.classic_dim_algs.np.concatenate")
    def test_concatenate_error(self, mock_concat, mock_path_mkdir, mock_np_load,
                               mock_makedirs, mock_s3_hook):
        """Тест: ошибка при склеивании массивов"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        mock_s3.list_keys.return_value = ["mri/processed/X_batch_1.npy"]

        mock_s3_conn = MagicMock()
        mock_s3.get_conn.return_value = mock_s3_conn
        mock_s3_conn.download_file.return_value = None

        # Мокаем успешную загрузку numpy
        X_mock = np.random.rand(10, 64, 64, 3)
        y_mock = np.random.randint(0, 2, 10)
        mock_np_load.side_effect = [X_mock, y_mock]

        # Ошибка при конкатенации
        mock_concat.side_effect = ValueError("need at least one array to concatenate")

        with pytest.raises(ValueError):
            load_data_from_s3("test-bucket", "mri", "test_dir")