from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import datetime
import pandas as pd

TMP_DIR = "/tmp/mri_batches"

def _final_check(ti):
    from scripts.dag_config import BUCKET_NAME

    train_tasks = [
        "train_svm_pca",
        "train_logreg_pca",
        "train_svm_umap",
        "train_logreg_umap",
        "train_svm_tda",
        "train_logreg_tda",
    ]

    # забираем XCom от всех train-тасков
    metrics_list = ti.xcom_pull(task_ids=train_tasks)
    metrics_list = [m for m in metrics_list if m is not None]

    if not metrics_list:
        raise ValueError("Ни одной метрики не было получено")

    # формируем таблицу
    df = pd.DataFrame(metrics_list)
    df = df.sort_values("f1", ascending=False)

    # дата запуска
    run_date = datetime.date.today().isoformat()

    run_dir = os.path.join(TMP_DIR, run_date)
    os.makedirs(run_dir, exist_ok=True)

    final_path = os.path.join(run_dir, "final_report.csv")

    df.to_csv(final_path, index=False)

    # загружаем в S3
    s3 = S3Hook(aws_conn_id="s3")
    execution_ts = ti.execution_date.strftime("%H-%M-%S")

    s3_key = f"reports/{run_date}/{execution_ts}.csv"

    s3.load_file(
        filename=final_path,
        key=s3_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    os.remove(final_path)

    return {
        "best_model": df.iloc[0].to_dict(),
        "report_s3_key": f"reports/{run_date}/final_report.csv",
    }