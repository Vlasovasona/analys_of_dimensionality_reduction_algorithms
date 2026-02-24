import numpy as np
from unittest.mock import patch, MagicMock


@patch("scripts.data_extraction.mri.kagglehub.dataset_download")
@patch("scripts.data_extraction.mri.os.listdir")
@patch("scripts.data_extraction.mri.os.makedirs")
@patch("scripts.data_extraction.mri.shutil.copytree")
@patch("scripts.data_extraction.mri.shutil.copy2")
@patch("scripts.data_extraction.mri.os.path.isdir")
def test_download_mri_dataset(
    mock_isdir,
    mock_copy2,
    mock_copytree,
    mock_makedirs,
    mock_listdir,
    mock_kaggle,
):
    mock_kaggle.return_value = "/fake/kaggle/path"
    mock_listdir.return_value = ["class1", "readme.txt"]
    mock_isdir.side_effect = [True, False]

    from scripts.data_extraction.mri import _download_mri_dataset
    _download_mri_dataset()

    mock_kaggle.assert_called_once()
    mock_makedirs.assert_called_once()
    mock_copytree.assert_called_once()
    mock_copy2.assert_called_once()


@patch("scripts.data_extraction.mri.os.walk")
@patch("scripts.data_extraction.mri.S3Hook")
def test_upload_images_to_s3(mock_s3, mock_walk):
    mock_walk.return_value = [
        ("/data", [], ["img1.png", "img2.jpg"]),
    ]

    from scripts.data_extraction.mri import _upload_images_to_s3
    _upload_images_to_s3()

    s3 = mock_s3.return_value
    assert s3.load_file.call_count == 2


@patch("scripts.data_extraction.mri.os.remove")
@patch("scripts.data_extraction.mri.np.save")
def test_save_batch(mock_save, mock_remove):
    s3 = MagicMock()

    X = [np.zeros((64, 64, 3))]
    y = [1]

    from scripts.data_extraction.mri import _save_batch
    _save_batch(X, y, batch_id=1, s3=s3)

    assert mock_save.call_count == 2
    assert s3.load_file.call_count == 2
    assert mock_remove.call_count == 2


@patch("scripts.data_extraction.mri.os.remove")
@patch("scripts.data_extraction.mri.np.save")
def test_save_batch_tda(mock_save, mock_remove):
    s3 = MagicMock()

    X = [np.zeros((64, 64, 3))]
    y = [0]

    from scripts.data_extraction.mri import _save_batch_tda
    _save_batch_tda(X, y, batch_id=3, s3=s3)

    assert s3.load_file.call_count == 2


@patch("cv2.cvtColor")
@patch("cv2.resize")
@patch("cv2.Sobel")
@patch("cv2.GaussianBlur")
def test_preprocess_image(
    mock_gaussian,
    mock_sobel,
    mock_resize,
    mock_cvt,
):
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    mock_cvt.return_value = np.ones((100, 100))
    mock_resize.return_value = np.ones((64, 64))
    mock_sobel.return_value = np.ones((64, 64))
    mock_gaussian.return_value = np.ones((64, 64))

    from scripts.data_extraction.mri import preprocess_image
    out = preprocess_image(img, 64)

    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float64

@patch("scripts.data_extraction.mri._save_batch")
@patch("scripts.data_extraction.mri.S3Hook")
@patch("cv2.imdecode")
@patch("cv2.cvtColor")
@patch("cv2.resize")
def test_preprocess_mri_images(
    mock_resize,
    mock_cvt,
    mock_imdecode,
    mock_s3,
    mock_save,
):
    mock_s3.return_value.list_keys.return_value = [
        "mri/raw/class1/img1.png",
        "mri/raw/class2/img2.png",
    ]

    mock_imdecode.return_value = np.ones((100, 100, 3), dtype=np.uint8)
    mock_cvt.return_value = np.ones((100, 100, 3), dtype=np.uint8)
    mock_resize.return_value = np.ones((64, 64, 3), dtype=np.uint8)

    from scripts.data_extraction.mri import _preprocess_mri_images
    _preprocess_mri_images()

    assert mock_save.called


@patch("scripts.data_extraction.mri._save_batch_tda")
@patch("scripts.data_extraction.mri.preprocess_image")
@patch("scripts.data_extraction.mri.S3Hook")
@patch("cv2.imdecode")
def test_preprocess_mri_images_to_tda(
    mock_imdecode,
    mock_s3,
    mock_preprocess,
    mock_save,
):
    mock_s3.return_value.list_keys.return_value = [
        "mri/raw/class1/img1.png",
    ]

    mock_imdecode.return_value = np.ones((100, 100, 3), dtype=np.uint8)
    mock_preprocess.return_value = np.ones((64, 64, 3))

    from scripts.data_extraction.mri import _preprocess_mri_images_to_tda
    _preprocess_mri_images_to_tda()

    mock_save.assert_called()