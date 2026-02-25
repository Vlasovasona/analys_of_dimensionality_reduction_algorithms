import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from scripts.train_models import (
    load_dim_data_from_s3,
    _train_model,
)

@patch("scripts.train_models.np.load")
@patch("scripts.train_models.S3Hook")
def test_load_dim_data_success(mock_s3, mock_np_load):
    X_train = np.random.rand(10, 5)
    X_test = np.random.rand(4, 5)
    y_train = np.array([0, 1] * 5)
    y_test = np.array([1, 0, 1, 0])

    mock_np_load.side_effect = [X_train, X_test, y_train, y_test]

    result = load_dim_data_from_s3(
        bucket_name="bucket",
        processed_prefix="prefix",
        local_data_dir="data",
        dim_alg_name="pca",
    )

    assert len(result) == 4
    assert result[0].shape == (10, 5)


@patch("scripts.train_models.S3Hook")
def test_load_dim_data_missing_X(mock_s3):
    mock_s3.return_value.get_conn.return_value.download_file.side_effect = Exception("S3 error")

    with pytest.raises(Exception):
        load_dim_data_from_s3()


@patch("scripts.train_models.np.load")
@patch("scripts.train_models.S3Hook")
def test_load_dim_data_missing_y(mock_s3, mock_np_load):
    X_train = np.random.rand(5, 3)
    X_test = np.random.rand(2, 3)

    mock_np_load.side_effect = [X_train, X_test, Exception("no y")]

    with pytest.raises(Exception):
        load_dim_data_from_s3()


@pytest.fixture
def fake_data():
    return (
        np.random.rand(20, 5),
        np.random.rand(5, 5),
        np.random.randint(0, 2, 20),
        np.random.randint(0, 2, 5),
    )

@patch("scripts.train_models.mlflow")
@patch("scripts.train_models.load_dim_data_from_s3")
@patch("scripts.train_models.GridSearchCV")
def test_train_model_logreg_success(
    mock_grid,
    mock_load,
    mock_mlflow,
    fake_data,
):
    mock_load.return_value = fake_data

    mock_estimator = MagicMock()
    mock_estimator.predict.return_value = fake_data[3]

    mock_grid.return_value.fit.return_value = None
    mock_grid.return_value.best_estimator_ = mock_estimator
    mock_grid.return_value.best_params_ = {"clf__C": 1.0}

    metrics = _train_model(
        dimensionally_alg_type="pca",
        model_type="logreg",
    )

    assert metrics["alg_name"] == "pca/logreg"
    assert "accuracy" in metrics
    mock_mlflow.log_metric.assert_called()


@patch("scripts.train_models.mlflow")
@patch("scripts.train_models.load_dim_data_from_s3")
@patch("scripts.train_models.GridSearchCV")
def test_train_model_svm_success(
    mock_grid,
    mock_load,
    mock_mlflow,
    fake_data,
):
    mock_load.return_value = fake_data

    mock_estimator = MagicMock()
    mock_estimator.predict.return_value = fake_data[3]

    mock_grid.return_value.fit.return_value = None
    mock_grid.return_value.best_estimator_ = mock_estimator
    mock_grid.return_value.best_params_ = {"clf__C": 10}

    metrics = _train_model("pca", "svm")

    assert metrics["alg_name"] == "pca/svm"


@patch("scripts.train_models.load_dim_data_from_s3")
@patch("scripts.train_models.mlflow")
def test_train_model_unknown_type(mock_mlflow, mock_load):
    mock_load.return_value = (
        np.zeros((1, 2)),  # X_train
        np.zeros((1, 2)),  # X_test
        np.zeros(1),       # y_train
        np.zeros(1),       # y_test
    )

    with pytest.raises(ValueError, match="Unknown model type"):
        _train_model(
            dimensionally_alg_type="pca",
            model_type="knn",
        )