from airflow import DAG
from airflow.operators.python import PythonOperator

# from airflow.utils.dates import days_ago
import pendulum
from scripts.parameters_validation.validate_classic_dim_algs import validate_dimensionality_config
from scripts.parameters_validation.dag_config_validation import validate_storage_config
from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR
from scripts.dim_algs_config import DIM_ALGORITHMS

# Wrapper-функции для облегчения импортов при инициализации DAG scheduler'ом

def download_mri_dataset_callable(**context):
    from scripts.data_extraction.mri import _download_mri_dataset
    return _download_mri_dataset()


def upload_images_to_s3_callable(**context):
    from scripts.data_extraction.mri import _upload_images_to_s3
    return _upload_images_to_s3()


def preprocess_mri_images_callable(**context):
    from scripts.data_extraction.mri import _preprocess_mri_images
    return _preprocess_mri_images()


def preprocess_mri_images_tda_callable(**context):
    from scripts.data_extraction.mri import _preprocess_mri_images_to_tda
    return _preprocess_mri_images_to_tda()


def train_dim_model_callable(**context):
    from scripts.classic_dim_algs import _train_dim_model
    return _train_dim_model()


def prepare_train_test_datasets_callable(**context):
    from scripts.classic_dim_algs import _prepare_train_test_datasets
    return _prepare_train_test_datasets()


def train_models_callable(**context):
    from scripts.train_models import _train_model
    return _train_model()


def compute_persistence_diagrams_callable(**context):
    from scripts.TDA.tda_dag_functions import _compute_persistence_diagrams
    return _compute_persistence_diagrams()


def vectorize_persistence_diagrams_callable(**context):
    from scripts.TDA.tda_dag_functions import _vectorize_persistence_diagrams
    return _vectorize_persistence_diagrams(op_kwargs={
            "test_size": 0.2,
        })

# Закончен блок wrapper-функций

def build_dim_tasks(
    alg_name: str,
    hyperparams: dict,
    preprocess_task,
    dag,
):
    """Построитель тасков по обучению классических алгоритмов понижения размерности"""
    validate_task = PythonOperator(
        task_id=f"validate_{alg_name}_config",
        python_callable=validate_dimensionality_config,
        op_kwargs={
            "dimensionally_alg_type": alg_name,
            "dim_arg_hyperparams": hyperparams,
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
        },
        dag=dag,
    )

    train_task = PythonOperator(
        task_id=f"create_{alg_name}",
        python_callable=train_dim_model_callable,
        op_kwargs={
            "dimensionally_alg_type": alg_name,
            "dim_arg_hyperparams": hyperparams,
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
        },
        dag=dag,
    )

    preprocess_task >> validate_task >> train_task
    return train_task

# Создание DAG

dag = DAG(
    dag_id="dimensionality_reduction_algorithms",
    start_date=pendulum.datetime(2026, 1, 1),
    catchup=False,
)

with dag:

    validate_dag_config = PythonOperator(
        task_id="validate_dag_config",
        python_callable=validate_storage_config,
    )

    download_mri_dataset = PythonOperator(
        task_id="download_mri_dataset",
        python_callable=download_mri_dataset_callable,
    )

    upload_images_to_s3 = PythonOperator(
        task_id="upload_images_to_s3",
        python_callable=upload_images_to_s3_callable,
    )

    preprocess_mri_images = PythonOperator(
        task_id="preprocess_mri_images",
        python_callable=preprocess_mri_images_callable,
    )

    preprocess_mri_images_tda = PythonOperator(
        task_id="preprocess_mri_images_tda",
        python_callable=preprocess_mri_images_tda_callable,
    )

    prepare_train_test = PythonOperator(
        task_id="prepare_train_test_datasets",
        python_callable=prepare_train_test_datasets_callable,
        op_kwargs={
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
        },
    )

    train_dim_tasks = {}

    for alg_name, hyperparams in DIM_ALGORITHMS.items():
        train_dim_tasks[alg_name] = build_dim_tasks(
            dag=dag,
            alg_name=alg_name,
            hyperparams=hyperparams,
            preprocess_task=prepare_train_test,
        )


    train_logreg_pca = PythonOperator(
        task_id="train_logreg_pca",
        python_callable=train_models_callable,
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
        python_callable=train_models_callable,
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

    train_logreg_umap = PythonOperator(
        task_id="train_logreg_umap",
        python_callable=train_models_callable,
        op_kwargs={
            "dimensionally_alg_type": "umap",
            "model_type": "logreg",
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
            "mlflow_experiment_name": "mri-brain-tumor",
            "mlflow_uri": "http://mlflow:5000",
        },
    )

    train_svm_umap = PythonOperator(
        task_id="train_svm_umap",
        python_callable=train_models_callable,
        op_kwargs={
            "dimensionally_alg_type": "umap",
            "model_type": "svm",
            "bucket_name": BUCKET_NAME,
            "processed_prefix": PROCESSED_PREFIX,
            "local_data_dir": LOCAL_DATA_DIR,
            "mlflow_experiment_name": "mri-brain-tumor",
            "mlflow_uri": "http://mlflow:5000",
        },
    )

    compute_persistence_diagrams = PythonOperator(
        task_id="compute_persistence_diagrams",
        python_callable=compute_persistence_diagrams_callable,
    )

    vectorize_persistence_diagrams = PythonOperator(
        task_id="vectorize_persistence_diagrams",
        python_callable=vectorize_persistence_diagrams_callable,
    )

    validate_dag_config >> download_mri_dataset >> upload_images_to_s3

    upload_images_to_s3 >> preprocess_mri_images
    upload_images_to_s3 >> preprocess_mri_images_tda

    preprocess_mri_images_tda >> compute_persistence_diagrams >> vectorize_persistence_diagrams

    preprocess_mri_images >> prepare_train_test

    train_dim_tasks["pca"] >> [train_logreg_pca, train_svm_pca]
    train_dim_tasks["umap"] >> [train_logreg_umap, train_svm_umap]
