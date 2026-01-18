ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.7.3-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

USER airflow
RUN pip install --no-cache-dir kagglehub

USER root
RUN mkdir -p /opt/airflow/data \
    && chown airflow /opt/airflow/data

USER airflow
