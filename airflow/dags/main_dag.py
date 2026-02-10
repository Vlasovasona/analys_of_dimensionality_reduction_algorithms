from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from scripts.mri_loading_and_processing import _download_mri_dataset, _upload_images_to_s3, _preprocess_mri_images
from scripts.train_models import _train_model
from scripts.classic_dim_algs import _train_dim_model

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
    start_date=days_ago(1),
    schedule_interval=None,
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
            "bucket_name": "mri-dataset",
            "processed_prefix": 'mri',
            "local_data_dir": "mri_train_data",
        },
    )

    create_tsne = PythonOperator(
        task_id="create_tsne",
        python_callable=_train_dim_model,
        op_kwargs={
            "dimensionally_alg_type": "tsne",
            "dim_arg_hyperparams": tsne_dict,
            "bucket_name": "mri-dataset",
            "processed_prefix": 'mri',
            "local_data_dir": "mri_train_data",
        },
    )

    create_umap = PythonOperator(
        task_id="create_umap",
        python_callable=_train_dim_model,
        op_kwargs={
            "dimensionally_alg_type": "umap",
            "dim_arg_hyperparams": umap_dict,
            "bucket_name": "mri-dataset",
            "processed_prefix": 'mri',
            "local_data_dir": "mri_train_data",
        },
    )

    # download_mri_dataset >> upload_images_to_s3 >> preprocess_mri_images >> [create_pca >> [train_logreg, train_svm], create_tsne >>  [train_logreg, train_svm], create_umap >> [train_logreg, train_svm], create_TDA >>  [train_logreg, train_svm], train_CNN]

    download_mri_dataset >> upload_images_to_s3 >> preprocess_mri_images

    preprocess_mri_images >> [
        create_pca,
        create_tsne,
        create_umap,
        # create_TDA,
        # train_CNN
    ]

    # create_pca >> [train_logreg, train_svm]
    # create_tsne >> [train_logreg, train_svm]
    # create_umap >> [train_logreg, train_svm]
    # create_TDA >> [train_logreg, train_svm]