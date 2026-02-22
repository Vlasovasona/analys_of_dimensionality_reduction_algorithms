from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

# Wrapper functions

def validate_dag_config_callable(**_):
    from scripts.parameters_validation.dag_config_validation import validate_storage_config
    from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR

    return validate_storage_config(
        bucket_name=BUCKET_NAME,
        processed_prefix=PROCESSED_PREFIX,
        local_data_dir=LOCAL_DATA_DIR,
    )


def download_mri_dataset_callable(**_):
    from scripts.data_extraction.mri import _download_mri_dataset
    return _download_mri_dataset()


def upload_images_to_s3_callable(**_):
    from scripts.data_extraction.mri import _upload_images_to_s3
    return _upload_images_to_s3()


def preprocess_mri_images_callable(**_):
    from scripts.data_extraction.mri import _preprocess_mri_images
    return _preprocess_mri_images()


def preprocess_mri_images_tda_callable(**_):
    from scripts.data_extraction.mri import _preprocess_mri_images_to_tda
    return _preprocess_mri_images_to_tda()


def prepare_train_test_datasets_callable(**_):
    from scripts.classic_dim_algs import _prepare_train_test_datasets
    from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR

    return _prepare_train_test_datasets(
        bucket_name=BUCKET_NAME,
        processed_prefix=PROCESSED_PREFIX,
        local_data_dir=LOCAL_DATA_DIR,
    )


def validate_dimensionality_config_callable(alg_name, hyperparams, **_):
    from scripts.parameters_validation.validate_classic_dim_algs import validate_dimensionality_config
    from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR

    return validate_dimensionality_config(
        dimensionally_alg_type=alg_name,
        dim_arg_hyperparams=hyperparams,
        bucket_name=BUCKET_NAME,
        processed_prefix=PROCESSED_PREFIX,
        local_data_dir=LOCAL_DATA_DIR,
    )


def train_dim_model_callable(alg_name, hyperparams, **_):
    from scripts.classic_dim_algs import _train_dim_model
    from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR

    return _train_dim_model(
        dimensionally_alg_type=alg_name,
        dim_arg_hyperparams=hyperparams,
        bucket_name=BUCKET_NAME,
        processed_prefix=PROCESSED_PREFIX,
        local_data_dir=LOCAL_DATA_DIR,
    )


def train_model_callable(dim_alg, model_type, **_):
    from scripts.train_models import _train_model
    from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR

    return _train_model(
        dimensionally_alg_type=dim_alg,
        model_type=model_type,
        bucket_name=BUCKET_NAME,
        processed_prefix=PROCESSED_PREFIX,
        local_data_dir=LOCAL_DATA_DIR,
        mlflow_experiment_name="mri-brain-tumor",
        mlflow_uri="http://mlflow:5000",
    )


def compute_persistence_diagrams_callable(**_):
    from scripts.TDA.tda_dag_functions import _compute_persistence_diagrams
    return _compute_persistence_diagrams()


def vectorize_persistence_diagrams_callable(**_):
    from scripts.TDA.tda_dag_functions import _vectorize_persistence_diagrams
    return _vectorize_persistence_diagrams(test_size=0.2)


def prepare_train_test_datasets_tda_callable(**_):
    from scripts.TDA.tda_dag_functions import _prepare_train_test_datasets_tda
    from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR

    return _prepare_train_test_datasets_tda(
        bucket_name=BUCKET_NAME,
        processed_prefix=PROCESSED_PREFIX,
        local_data_dir=LOCAL_DATA_DIR,
        test_size=0.2,
        random_state=42,
    )


def train_tda_model_callable(model_type, **_):
    from scripts.TDA.tda_dag_functions import _train_TDA_models
    from scripts.dag_config import BUCKET_NAME, PROCESSED_PREFIX, LOCAL_DATA_DIR

    return _train_TDA_models(
        model_type=model_type,
        bucket_name=BUCKET_NAME,
        processed_prefix=PROCESSED_PREFIX,
        local_data_dir=LOCAL_DATA_DIR,
        mlflow_experiment_name="mri-brain-tumor",
        mlflow_uri="http://mlflow:5000",
    )


# DAG

dag = DAG(
    dag_id="dimensionality_reduction_algorithms",
    start_date=pendulum.datetime(2026, 1, 1),
    catchup=False,
)

with dag:

    validate_dag_config = PythonOperator(
        task_id="validate_dag_config",
        python_callable=validate_dag_config_callable,
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
    )

    from scripts.dim_algs_config import DIM_ALGORITHMS

    train_dim_tasks = {}

    for alg_name, hyperparams in DIM_ALGORITHMS.items():
        validate_task = PythonOperator(
            task_id=f"validate_{alg_name}_config",
            python_callable=validate_dimensionality_config_callable,
            op_kwargs={"alg_name": alg_name, "hyperparams": hyperparams},
        )

        train_task = PythonOperator(
            task_id=f"train_{alg_name}",
            python_callable=train_dim_model_callable,
            op_kwargs={"alg_name": alg_name, "hyperparams": hyperparams},
        )

        prepare_train_test >> validate_task >> train_task
        train_dim_tasks[alg_name] = train_task

    # ML models
    train_dim_tasks["pca"] >> [
        PythonOperator(
            task_id="train_logreg_pca",
            python_callable=train_model_callable,
            op_kwargs={"dim_alg": "pca", "model_type": "logreg"},
        ),
        PythonOperator(
            task_id="train_svm_pca",
            python_callable=train_model_callable,
            op_kwargs={"dim_alg": "pca", "model_type": "svm"},
        ),
    ]

    train_dim_tasks["umap"] >> [
        PythonOperator(
            task_id="train_logreg_umap",
            python_callable=train_model_callable,
            op_kwargs={"dim_alg": "umap", "model_type": "logreg"},
        ),
        PythonOperator(
            task_id="train_svm_umap",
            python_callable=train_model_callable,
            op_kwargs={"dim_alg": "umap", "model_type": "svm"},
        ),
    ]

    compute_pd = PythonOperator(
        task_id="compute_persistence_diagrams",
        python_callable=compute_persistence_diagrams_callable,
    )

    vectorize_pd = PythonOperator(
        task_id="vectorize_persistence_diagrams",
        python_callable=vectorize_persistence_diagrams_callable,
    )

    prepare_tda = PythonOperator(
        task_id="prepare_train_test_datasets_tda",
        python_callable=prepare_train_test_datasets_tda_callable,
    )

    train_tda = [
        PythonOperator(
            task_id="train_logreg_tda",
            python_callable=train_tda_model_callable,
            op_kwargs={"model_type": "logreg"},
        ),
        PythonOperator(
            task_id="train_svm_tda",
            python_callable=train_tda_model_callable,
            op_kwargs={"model_type": "svm"},
        ),
    ]

    validate_dag_config >> download_mri_dataset >> upload_images_to_s3
    upload_images_to_s3 >> preprocess_mri_images >> prepare_train_test
    upload_images_to_s3 >> preprocess_mri_images_tda >> compute_pd >> vectorize_pd >> prepare_tda >> train_tda