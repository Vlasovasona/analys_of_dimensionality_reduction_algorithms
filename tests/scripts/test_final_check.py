import os
import pytest
from unittest.mock import MagicMock, patch
from scripts.final_check import _final_check

@pytest.fixture
def mock_ti():
    """Мокаем объект TaskInstance с XCom"""
    ti = MagicMock()
    ti.xcom_pull.return_value = [
        {"alg_name": "pca/logreg", "f1": 0.8},
        {"alg_name": "pca/svm", "f1": 0.85},
        {"alg_name": "umap/logreg", "f1": 0.75},
        None
    ]
    ti.execution_date.strftime.return_value = "12-00-00"
    return ti

@patch("scripts.final_check.S3Hook")
@patch("scripts.final_check.pd.DataFrame.to_csv")
@patch("scripts.final_check.os.remove")
@patch("scripts.final_check.os.makedirs")
def test_final_check_success(mock_makedirs, mock_remove, mock_to_csv, mock_s3hook, mock_ti):
    mock_s3 = mock_s3hook.return_value
    mock_s3.load_file.return_value = True

    result = _final_check(mock_ti)

    expected_tasks = [
        "train_svm_pca",
        "train_logreg_pca",
        "train_svm_umap",
        "train_logreg_umap",
        "train_svm_tda",
        "train_logreg_tda",
    ]
    mock_ti.xcom_pull.assert_called_once_with(task_ids=expected_tasks)

    mock_makedirs.assert_called()
    mock_to_csv.assert_called()
    mock_remove.assert_called()

    mock_s3.load_file.assert_called_once()
    args, kwargs = mock_s3.load_file.call_args
    assert kwargs["bucket_name"]
    assert kwargs["replace"] is True

    assert "best_model" in result
    assert result["best_model"]["f1"] == 0.85
    assert "report_s3_key" in result