from airflow import DAG
from airflow.operators.python import PythonOperator
# from airflow.utils.dates import days_ago
import pendulum
from scripts.data_extraction import _download_mri_dataset, _upload_images_to_s3, _preprocess_mri_images
from scripts.train_models import _train_model
from scripts.classic_dim_algs import _train_dim_model, _load_and_concat_targets_from_s3

BUCKET_NAME = "mri-dataset"
PROCESSED_PREFIX = "mri"
LOCAL_DATA_DIR = "mri_train_data"

pca_dict = {
    "pca_components": 120,
}

tsne_dict = {
    "n_components": 2,
    "perplexity": 30,
    "early_exaggeration": 12,
    "learning_rate": "auto",
    "metric": "euclidean",
    "init": "pca",
    "angle": 0.5,
}

umap_dict = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "euclidean",
    "spread": 1.0,
    "low_memory": True,
    "init": "spectral",
}

dag = DAG(
    dag_id='dimensionality_reduction_algorithms',
    start_date=pendulum.datetime(2026, 1, 1),
    # schedule_interval=None,
    catchup=False
)

with dag:

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

    create_pca = PythonOperator(
        task_id="create_pca",
        python_callable=_train_dim_model,
        op_kwargs={
            "dimensionally_alg_type": "pca",
            "dim_arg_hyperparams": pca_dict,
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
        },
    )

    create_tsne = PythonOperator(
        task_id="create_tsne",
        python_callable=_train_dim_model,
        op_kwargs={
            "dimensionally_alg_type": "tsne",
            "dim_arg_hyperparams": tsne_dict,
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
        },
    )

    create_umap = PythonOperator(
        task_id="create_umap",
        python_callable=_train_dim_model,
        op_kwargs={
            "dimensionally_alg_type": "umap",
            "dim_arg_hyperparams": umap_dict,
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
        },
    )

    concat_y = PythonOperator(
        task_id="concat_y",
        python_callable=_load_and_concat_targets_from_s3,
        op_kwargs={
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
        },
    )

    train_logreg_pca = PythonOperator(
        task_id="train_logreg_pca",
        python_callable=_train_model,
        op_kwargs={
            "dimensionally_alg_type": "pca",
            "model_type": "logreg",
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
            "mlflow_experiment_name": "mri-brain-tumor",
            "mlflow_uri": "http://mlflow:5000",
        },
    )

    train_svm_pca = PythonOperator(
        task_id="train_svm_pca",
        python_callable=_train_model,
        op_kwargs={
            "dimensionally_alg_type": "pca",
            "model_type": "svm",
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
            "mlflow_experiment_name": "mri-brain-tumor",
            "mlflow_uri": "http://mlflow:5000",
        },
    )

    download_mri_dataset >> upload_images_to_s3 >> preprocess_mri_images

    preprocess_mri_images >> [
        create_pca,
        create_tsne,
        create_umap,
        concat_y,
        # create_TDA,
        # train_CNN
    ]

    create_pca >> [train_logreg_pca, train_svm_pca]
    # create_tsne >> [train_logreg, train_svm]
    # create_umap >> [train_logreg, train_svm]
    # create_TDA >> [train_logreg, train_svm]