import numpy as np
import pickle
import pytest
from unittest.mock import patch, MagicMock

from scripts.TDA.tda_dag_functions import (
    _compute_persistence_diagrams,
    _prepare_train_test_datasets_tda,
    _vectorize_persistence_diagrams,
    load_TDA_data_from_s3,
    _train_TDA_models
)


@patch("scripts.TDA.tda_dag_functions.validate_loaded_arrays")
@patch("scripts.TDA.tda_dag_functions.S3Hook")
@patch("scripts.TDA.tda_dag_functions.np.load")
def test_load_tda_data_success(mock_np_load, mock_s3hook, mock_validate):
    mock_s3 = mock_s3hook.return_value
    mock_s3.get_conn.return_value.download_file.return_value = None

    mock_np_load.side_effect = [
        np.zeros((5, 10)),
        np.zeros((2, 10)),
        np.zeros(5),
        np.zeros(2),
    ]

    X_train, X_test, y_train, y_test = load_TDA_data_from_s3()

    assert X_train.shape[0] == 5
    assert X_test.shape[0] == 2
    mock_validate.assert_called()


@patch("scripts.TDA.tda_dag_functions.load_TDA_data_from_s3")
@patch("scripts.TDA.tda_dag_functions.mlflow")
def test_train_tda_models_unknown_type(mock_mlflow, mock_load):
    mock_load.return_value = (
        np.zeros((1, 2)),  # X_train
        np.zeros((1, 2)),  # X_test
        np.zeros(1),       # y_train
        np.zeros(1),       # y_test
    )

    mock_mlflow.start_run.return_value.__enter__.return_value = None

    with pytest.raises(ValueError, match="Unknown model type"):
        _train_TDA_models(model_type="knn")


@patch("scripts.TDA.tda_dag_functions.S3Hook")
@patch("scripts.TDA.tda_dag_functions.np.load")
@patch("scripts.TDA.tda_dag_functions.np.save")
def test_prepare_train_test_datasets_tda_success(
    mock_save,
    mock_load,
    mock_s3hook,
):
    mock_s3 = mock_s3hook.return_value
    mock_s3.list_keys.return_value = [
        "mri/processed/TDA/y_batch1.npy"
    ]

    mock_load.side_effect = [
        np.array([0, 1] * 10),  # y: 20 элементов
        np.random.rand(20, 10),  # X
    ]

    _prepare_train_test_datasets_tda(
        bucket_name="mri-dataset",
        processed_prefix="mri",
        local_data_dir="tmp",
    )

    assert mock_save.call_count == 4
    assert mock_s3.load_file.call_count == 4


