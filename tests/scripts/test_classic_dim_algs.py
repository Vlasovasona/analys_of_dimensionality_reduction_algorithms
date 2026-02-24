import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from scripts.classic_dim_algs import _train_dim_model, _prepare_train_test_datasets, load_train_test_from_s3


class TestTrainDimModel:

    def setup_method(self):
        self.X_train = np.random.rand(10, 5)
        self.X_test = np.random.rand(5, 5)
        self.y_train = np.random.randint(0, 2, size=10)
        self.y_test = np.random.randint(0, 2, size=5)


    @patch("scripts.classic_dim_algs.Path.unlink")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.mlflow")
    @patch("scripts.classic_dim_algs.PCA")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.load_train_test_from_s3")
    def test_train_dim_model_pca_success(
        self,
        mock_load,
        mock_validate,
        mock_pca,
        mock_mlflow,
        mock_s3,
        mock_save,
        mock_unlink,
    ):
        mock_load.return_value = (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = self.X_train
        mock_model.transform.return_value = self.X_test
        mock_model.explained_variance_ratio_ = [0.5, 0.3]
        mock_pca.return_value = mock_model

        _train_dim_model(
            "pca",
            {"pca_components": 2},
            "bucket",
            "prefix",
            "tmp",
        )

        mock_validate.assert_called_once_with(self.X_train, self.y_train)
        mock_pca.assert_called_once_with(n_components=2)
        mock_mlflow.start_run.assert_called_once()
        mock_save.assert_called()
        mock_s3.return_value.load_file.assert_called()
        mock_unlink.assert_called()


    @patch("scripts.classic_dim_algs.Path.unlink")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.mlflow")
    @patch("scripts.classic_dim_algs.UMAP")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.load_train_test_from_s3")
    def test_train_dim_model_umap_success(
        self,
        mock_load,
        mock_validate,
        mock_umap,
        mock_mlflow,
        mock_s3,
        mock_save,
        mock_unlink,
    ):
        mock_load.return_value = (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = self.X_train
        mock_model.transform.return_value = self.X_test
        mock_umap.return_value = mock_model

        params = {"n_neighbors": 5, "n_components": 2}

        _train_dim_model("umap", params, "bucket", "prefix", "tmp")

        mock_umap.assert_called_once_with(**params)
        mock_save.assert_called()
        mock_s3.return_value.load_file.assert_called()


    @patch("scripts.classic_dim_algs.Path.unlink")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    @patch("scripts.classic_dim_algs.logger")
    @patch("scripts.classic_dim_algs.PCA")
    @patch("scripts.classic_dim_algs.load_train_test_from_s3")
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.mlflow")
    def test_pca_components_over_max(
            self,
            mock_mlflow,
            mock_validate,
            mock_load,
            mock_pca,
            mock_logger,
            mock_s3,
            mock_save,
            mock_unlink,
    ):
        mock_load.return_value = (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )

        mock_model = MagicMock()
        mock_model.fit_transform.return_value = self.X_train
        mock_model.transform.return_value = self.X_test
        mock_model.explained_variance_ratio_ = [1.0]
        mock_pca.return_value = mock_model

        _train_dim_model(
            "pca",
            {"pca_components": 100},
            "bucket",
            "prefix",
            "tmp",
        )

        mock_pca.assert_called_once_with(n_components=5)
        mock_s3.return_value.load_file.assert_called()
        mock_unlink.assert_called()

class TestPrepareTrainTestDatasets:

    @patch("scripts.classic_dim_algs.train_test_split")
    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.np.save")
    @patch("scripts.classic_dim_algs.S3Hook")
    def test_prepare_train_test_success(
        self,
        mock_s3,
        mock_save,
        mock_load,
        mock_split,
    ):

        mock_s3.return_value.list_keys.return_value = [
            "prefix/processed/X_0000.npy",
            "prefix/processed/y_0000.npy",
        ]

        X = np.random.rand(4, 2)
        y = np.array([0, 1, 0, 1])

        mock_load.side_effect = [X, y]

        X_train, X_test, y_train, y_test = X[:3], X[3:], y[:3], y[3:]
        mock_split.return_value = (X_train, X_test, y_train, y_test)

        _prepare_train_test_datasets(
            bucket_name="bucket",
            processed_prefix="prefix",
            local_data_dir="data",
        )

        assert mock_save.call_count == 4
        assert mock_s3.return_value.load_file.call_count == 4

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_prepare_train_test_no_data(self, mock_s3):
        mock_s3.return_value.list_keys.return_value = []

        with pytest.raises(RuntimeError, match="Нет данных для split"):
            _prepare_train_test_datasets(
                bucket_name="bucket",
                processed_prefix="prefix",
                local_data_dir="data",
            )

class TestLoadTrainTestFromS3:
    @patch("scripts.classic_dim_algs.validate_loaded_arrays")
    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.S3Hook")
    def test_load_train_test_success(
            self,
            mock_s3,
            mock_load,
            mock_validate,
    ):
        X_train = np.random.rand(5, 2)
        X_test = np.random.rand(2, 2)
        y_train = np.array([0, 1, 0, 1, 0])
        y_test = np.array([1, 0])

        mock_load.side_effect = [
            X_train,
            X_test,
            y_train,
            y_test,
        ]

        Xtr, Xte, ytr, yte = load_train_test_from_s3(
            bucket_name="bucket",
            processed_prefix="prefix",
            local_data_dir="data",
        )

        assert Xtr.shape[0] == len(ytr)
        assert Xte.shape[0] == len(yte)
        assert mock_validate.call_count == 2

    @patch("scripts.classic_dim_algs.S3Hook")
    def test_load_train_test_no_credentials(self, mock_s3):
        mock_s3.side_effect = NoCredentialsError()

        with pytest.raises(ConnectionError, match="AWS credentials"):
            load_train_test_from_s3("bucket", "prefix", "data")

    @patch("scripts.classic_dim_algs.np.load")
    @patch("scripts.classic_dim_algs.S3Hook")
    def test_load_train_test_missing_file(
            self,
            mock_s3,
            mock_load,
    ):
        client_error = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "download_file",
        )

        mock_s3.return_value.get_conn.return_value.download_file.side_effect = client_error

        with pytest.raises(FileNotFoundError):
            load_train_test_from_s3("bucket", "prefix", "data")