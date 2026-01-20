ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.9.3-python3.10"
FROM ${AIRFLOW_BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER root
RUN mkdir -p /opt/airflow/data \
    && chown airflow: /opt/airflow/data

USER airflow
