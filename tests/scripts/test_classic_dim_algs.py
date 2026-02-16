import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from scripts.classic_dim_algs import load_data_from_s3, _load_and_concat_targets_from_s3, _train_dim_model


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
    def test_bucket_not_found(self, mock_s3_hook):
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

class TestLoadAndConcatTargetsFromS3:
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.Path.mkdir")
    @patch("scripts.classic_dim_algs.os.remove")
    def test_successful_load_and_upload(self, mock_remove, mock_path_mkdir, mock_np_save,
                                        mock_np_load, mock_makedirs, mock_s3_hook):
        """
        Тест: Успешная загрузка Y-файлов, конкатенация и загрузка в S3
        """
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        mock_s3.list_keys.return_value = [
            "mri/processed/X_batch_1.npy",
            "mri/processed/X_batch_2.npy",
            "mri/processed/y_batch_1.npy",
        ]

        mock_s3_conn = MagicMock()
        mock_s3.get_conn.return_value = mock_s3_conn
        mock_s3_conn.download_file.return_value = None

        y_mock_1 = np.random.randint(0, 2, 100)
        y_mock_2 = np.random.randint(0, 2, 150)
        mock_np_load.side_effect = [y_mock_1, y_mock_2]

        _load_and_concat_targets_from_s3(
            bucket_name="test-bucket",
            processed_prefix="mri",
            local_data_dir="test_dir"
        )

        mock_makedirs.assert_called_once_with("/tmp/test_dir", exist_ok=True)
        mock_s3.list_keys.assert_called_once_with("test-bucket", "mri/processed/")
        assert mock_s3_conn.download_file.call_count == 2

        expected_calls = [
            call(
                Bucket="test-bucket",
                Key="mri/processed/y_batch_1.npy",
                Filename=str(Path("/tmp/test_dir") / "y_batch_1.npy")
            ),
            call(
                Bucket="test-bucket",
                Key="mri/processed/y_batch_2.npy",
                Filename=str(Path("/tmp/test_dir") / "y_batch_2.npy")
            )
        ]
        mock_s3_conn.download_file.assert_has_calls(expected_calls, any_order=True)

        # проверяем сохранение конкатенированного массива
        mock_np_save.assert_called_once()
        args, _ = mock_np_save.call_args
        assert args[0] == "y_transformed.npy"
        assert args[1].shape == (250,)  # 100 + 150 = 250
        assert isinstance(args[1], np.ndarray)

        # проверяем загрузку в S3
        assert mock_s3.load_file.call_count == 1
        mock_s3.load_file.assert_called_once_with(
            filename="y_transformed.npy",
            key="mri/transformed/y_transformed.npy",
            bucket_name="test-bucket",
            replace=True
        )

        # проверяем удаление временного файла
        mock_remove.assert_called_once_with("y_transformed.npy")

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_no_credentials_error(self, mock_s3_hook):
        """Тест: нет AWS credentials -> ConnectionError"""
        mock_s3_hook.side_effect = NoCredentialsError()

        with pytest.raises(ConnectionError, match="Отсутствуют учетные данные AWS"):
            _load_and_concat_targets_from_s3(
                bucket_name="test-bucket",
                processed_prefix="mri",
                local_data_dir="test_dir"
            )

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_endpoint_connection_error(self, mock_s3_hook):
        """Тест: нет подключения к эндпоинту -> EndpointConnectionError"""
        mock_s3_hook.side_effect = EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")

        with pytest.raises(ConnectionError, match="Нет подключения к AWS эндпоинту"):
            _load_and_concat_targets_from_s3(
                bucket_name="test-bucket",
                processed_prefix="mri",
                local_data_dir="test_dir"
            )

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_bucket_not_exists(self, mock_s3_hook):
        """Тест: бакет не существует -> ValueError"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        error_response = {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket does not exist'}}
        mock_s3.list_keys.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(ValueError, match="Бакет test-bucket не найден"):
            _load_and_concat_targets_from_s3(
                bucket_name="test-bucket",
                processed_prefix="mri",
                local_data_dir="test_dir"
            )

    @patch("scripts.classic_dim_algs.os.makedirs")
    @patch("scripts.classic_dim_algs.S3Hook")
    def test_empty_bucket(self, mock_s3_hook, mock_makedirs):
        """Тест: если бакет пустой, должна вызываться ошибка RuntimeError"""

        mock_s3 = MagicMock()
        mock_s3.list_keys.return_value = []
        mock_s3_hook.return_value = mock_s3

        with pytest.raises(
                RuntimeError,
                match="В бакете test-bucket нет файлов по префиксу mri/processed/"
        ):
            _load_and_concat_targets_from_s3(
                bucket_name="test-bucket",
                processed_prefix="mri",
                local_data_dir="test_dir"
            )


    @patch("scripts.classic_dim_algs.S3Hook")
    def test_bucket_not_found(self, mock_s3_hook):
        """Тест: нет доступа к бакету -> PermissionError"""

        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3.list_keys.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(PermissionError, match="Нет доступа к бакету test-bucket"):
            _load_and_concat_targets_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    def test_other_client_error(self, mock_makedirs, mock_s3_hook):
        """Тест: другая ошибка ClientError -> ConnectionError"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        error_response = {'Error': {'Code': 'InternalError', 'Message': 'Internal error'}}
        mock_s3.list_keys.side_effect = ClientError(error_response, 'ListObjectsV2')

        with pytest.raises(ConnectionError, match="Ошибка S3"):
            _load_and_concat_targets_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    def test_list_keys_generic_error(self, mock_makedirs, mock_s3_hook):
        """Тест: общая ошибка при list_keys -> RuntimeError"""
        mock_s3 = MagicMock()
        mock_s3_hook.return_value = mock_s3

        mock_s3.list_keys.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Ошибка получения списка файлов: Network error"):
            _load_and_concat_targets_from_s3("test-bucket", "mri", "test_dir")

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
            _load_and_concat_targets_from_s3("test-bucket",
                                             "mri",
                                             "test_dir")

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
            _load_and_concat_targets_from_s3("test-bucket", "mri", "test_dir")

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
        y_mock_1 = np.random.randint(0, 2, 10)
        y_mock_2 = np.random.randint(0, 2, 20)
        mock_np_load.side_effect = [y_mock_1, y_mock_2]

        # Ошибка при конкатенации
        mock_concat.side_effect = ValueError("need at least one array to concatenate")

        with pytest.raises(ValueError):
            _load_and_concat_targets_from_s3("test-bucket", "mri", "test_dir")

    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.os.makedirs")
    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.Path.mkdir")
    def test_upload_to_s3_error(self, mock_path_mkdir, mock_np_save, mock_np_load,
                                mock_makedirs, mock_s3_hook):
        """Тест: Ошибка при загрузке файла в S3"""
        mock_s3_download = MagicMock()
        mock_s3_hook.return_value = mock_s3_download
        mock_s3_download.list_keys.return_value = ["mri/processed/X_batch_1.npy"]

        mock_s3_conn = MagicMock()
        mock_s3_download.get_conn.return_value = mock_s3_conn
        mock_s3_conn.download_file.return_value = None

        y_mock = np.random.randint(0, 2, 100)
        mock_np_load.return_value = y_mock

        mock_s3_upload = MagicMock()
        # Первый вызов возвращает mock_s3_download, второй - mock_s3_upload
        mock_s3_hook.side_effect = [mock_s3_download, mock_s3_upload]

        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3_upload.load_file.side_effect = ClientError(error_response, 'PutObject')

        with pytest.raises(ClientError):
            _load_and_concat_targets_from_s3("test-bucket", "mri", "test_dir")

class TestTrainDimModel:
    def setup_method(self):
        self.FAKE_X = np.random.rand(10, 5)
        self.FAKE_Y = np.random.randint(0, 2, size=10)

    @patch("scripts.classic_dim_algs.os.remove")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.mlflow")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.load_data_from_s3")
    def test_train_dim_model_pca_success(
            self,
            mock_load_data,
            mock_validate,
            mock_mlflow,
            mock_s3_hook,
            mock_save,
            mock_remove,
    ):
        """Тест: успешное обучение pca алгоритма"""
        mock_load_data.return_value = (self.FAKE_X, self.FAKE_Y)

        _train_dim_model(
            dimensionally_alg_type="pca",
            dim_arg_hyperparams={"pca_components": 2},
            bucket_name="test-bucket",
            processed_prefix="mri",
            local_data_dir="test_dir",
            mlflow_experiment_name="test-exp",
            mlflow_uri="http://mlflow",
        )

        mock_load_data.assert_called_once()
        mock_validate.assert_called_once_with(self.FAKE_X, self.FAKE_Y)
        mock_mlflow.start_run.assert_called_once()
        mock_save.assert_called_once()
        mock_s3_hook.return_value.load_file.assert_called_once()
        mock_remove.assert_called_once()

    @patch("scripts.classic_dim_algs.os.remove")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.mlflow")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.load_data_from_s3")
    def test_train_dim_model_tsne_success(
            self,
            mock_load_data,
            mock_validate,
            mock_mlflow,
            mock_s3_hook,
            mock_save,
            mock_remove,
    ):
        """Тест: успешное обучение t-SNE"""
        mock_load_data.return_value = (self.FAKE_X, self.FAKE_Y)

        tsne_params = {
            "n_components": 2,
            "perplexity": 5,
            "early_exaggeration": 12,
            "learning_rate": "auto",
        }

        _train_dim_model(
            dimensionally_alg_type="tsne",
            dim_arg_hyperparams=tsne_params,
            bucket_name="test-bucket",
            processed_prefix="mri",
            local_data_dir="test_dir",
            mlflow_experiment_name="test-exp",
            mlflow_uri="http://mlflow",
        )

        mock_save.assert_called_once()
        mock_s3_hook.return_value.load_file.assert_called_once()

    @patch("scripts.classic_dim_algs.TSNE")
    @patch("scripts.classic_dim_algs.load_data_from_s3")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    def test_train_dim_model_tsne_runtime_error(
            self,
            mock_validate,
            mock_load_data,
            mock_tsne,
    ):
        """Тест: ошибка обучения t-SNE"""
        mock_load_data.return_value = (self.FAKE_X, self.FAKE_Y)
        mock_tsne.side_effect = Exception("Boom")

        with pytest.raises(RuntimeError, match="Ошибка обучения t-SNE"):
            _train_dim_model(
                "tsne",
                {
                    "n_components": 2,
                    "perplexity": 5,
                    "early_exaggeration": 12,
                    "learning_rate": "auto",
                },
                "bucket",
                "prefix",
                "dir",
                "exp",
                "uri",
            )

    @patch("scripts.classic_dim_algs.os.remove")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.mlflow")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.load_data_from_s3")
    def test_train_dim_model_umap_success(
            self,
            mock_load_data,
            mock_validate,
            mock_mlflow,
            mock_s3_hook,
            mock_save,
            mock_remove,
    ):
        """Тест: успешное обучение umap"""
        mock_load_data.return_value = (self.FAKE_X, self.FAKE_Y)

        umap_params = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "spread": 1.0,
            "low_memory": True,
            "init": "spectral",
        }

        _train_dim_model(
            dimensionally_alg_type="umap",
            dim_arg_hyperparams=umap_params,
            bucket_name="test-bucket",
            processed_prefix="mri",
            local_data_dir="test_dir",
            mlflow_experiment_name="test-exp",
            mlflow_uri="http://mlflow",
        )

        mock_save.assert_called_once()
        mock_s3_hook.return_value.load_file.assert_called_once()

    @patch("scripts.classic_dim_algs.UMAP")
    @patch("scripts.classic_dim_algs.load_data_from_s3")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    def test_train_dim_model_umap_runtime_error(
            self,
            mock_validate,
            mock_load_data,
            mock_tsne,
    ):
        """Тест: ошибка обучения umap"""
        mock_load_data.return_value = (self.FAKE_X, self.FAKE_Y)
        mock_tsne.side_effect = Exception("Boom")

        with pytest.raises(RuntimeError, match="Ошибка обучения UMAP"):
            _train_dim_model(
                "umap",
                {
                    "n_neighbors": 15,
                    "min_dist": 0.1,
                    "n_components": 2,
                    "metric": "euclidean",
                    "spread": 1.0,
                    "low_memory": True,
                    "init": "spectral",
                },
                "bucket",
                "prefix",
                "dir",
                "exp",
                "uri",
            )

    @patch("scripts.classic_dim_algs.os.remove")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.logger")
    @patch("scripts.classic_dim_algs.mlflow")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.load_data_from_s3")
    def test_mlflow_log_param_error_is_caught(
            self,
            mock_load_data,
            mock_validate,
            mock_mlflow,
            mock_logger,
            mock_s3_hook,
            mock_save,
            mock_remove,
    ):
        mock_load_data.return_value = (self.FAKE_X, self.FAKE_Y)

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        mock_mlflow.log_param.side_effect = [
            None,
            Exception("mlflow error"),
        ]

        # act
        _train_dim_model(
            dimensionally_alg_type="pca",
            dim_arg_hyperparams={"pca_components": 2},
            bucket_name="test-bucket",
            processed_prefix="mri",
            local_data_dir="test_dir",
            mlflow_experiment_name="test-exp",
            mlflow_uri="http://mlflow",
        )

        # assert
        mock_logger.warning.assert_any_call(
            "Не удалось залогировать параметр pca_components: mlflow error"
        )

        mock_save.assert_called_once()
        mock_s3_hook.return_value.load_file.assert_called_once()
        mock_remove.assert_called_once()

    @patch("scripts.classic_dim_algs.os.remove")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.logger")
    @patch("scripts.classic_dim_algs.mlflow")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.load_data_from_s3")
    def test_n_components_overweight_max(
            self,
            mock_load_data,
            mock_validate,
            mock_mlflow,
            mock_logger,
            mock_s3_hook,
            mock_save,
            mock_remove,
    ):
        mock_load_data.return_value = (self.FAKE_X, self.FAKE_Y)

        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None

        mock_mlflow.log_param.return_value = None

        _train_dim_model(
            dimensionally_alg_type="pca",
            dim_arg_hyperparams={"pca_components": 100},
            bucket_name="test-bucket",
            processed_prefix="mri",
            local_data_dir="test_dir",
            mlflow_experiment_name="test-exp",
            mlflow_uri="http://mlflow",
        )


        mock_logger.warning.assert_any_call(
            "pca_components=100 больше максимально возможного 5. Уменьшено до 5"
        )

        mock_mlflow.log_param.assert_any_call("pca_components", 5)

        mock_save.assert_called_once()
        mock_s3_hook.return_value.load_file.assert_called_once()
        mock_remove.assert_called_once()



