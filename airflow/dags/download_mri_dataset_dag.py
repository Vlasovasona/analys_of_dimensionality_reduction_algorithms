from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from scripts.mri_loading_and_processing import _download_mri_dataset, _upload_images_to_s3, _preprocess_mri_images
from scripts.train_models import _train_model


with DAG(
    dag_id="download_brain_tumor_mri_dataset",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["dataset", "kaggle", "mri"]
) as dag:

    download_mri_dataset = PythonOperator(
        task_id="download_mri_dataset",
        python_callable=_download_mri_dataset,
    )

    upload_images_to_s3 = PythonOperator(
        task_id="upload_images_to_s3",
        python_callable=_upload_images_to_s3,
    )

    preprocess_mri_images = PythonOperator(
        task_id="preprocess_mri_images",
        python_callable=_preprocess_mri_images,
    )

    train_logreg = PythonOperator(
        task_id="train_logreg",
        python_callable=_train_model,
        op_kwargs={
            "model_type": "logreg",
            "pca_components": 128,
        },
    )

    train_svm = PythonOperator(
        task_id="train_svm",
        python_callable=_train_model,
        op_kwargs={
            "model_type": "svm",
            "pca_components": 128,
        },
    )

    download_mri_dataset >> upload_images_to_s3 >> preprocess_mri_images >> [train_logreg, train_svm]