# Extend base image with gsutil (https://github.com/GoogleCloudPlatform/cloud-builders/tree/master/gsutil)
FROM gcr.io/cloud-builders/gsutil
# Base Python image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Authenticate with default GCP service account automatically
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY dtu_mlops_team94/ dtu_mlops_team94/
COPY data/ data/

# # [ONLY FOR LOCAL BUILD] Copy service account key to container - authentication to GCP
# COPY service-account-key.json ~/service-account-key.json
# ENV GOOGLE_APPLICATION_CREDENTIALS=~/service-account-key.json

WORKDIR /
# [ONLY FOR LOCAL BUILD] Use cache to speed up build
RUN --mount=type=cache,target=/root/.cache pip install dvc dvc[gs]
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
# [FOR CLOUD BUILD]:
# RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

EXPOSE $PORT

# CMD ["python", "-u", "dtu_mlops_team94/train_model.py"]

CMD exec uvicorn dtu_mlops_team94.app:app --workers 1 --host 0.0.0.0 --port $PORT
