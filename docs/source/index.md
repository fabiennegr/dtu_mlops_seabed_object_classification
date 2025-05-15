## Documentation

Documentation for dtu_mlops_team94


### Using Docker images (trainer)
To build the docker image, run:
```sh
docker build -t trainer:latest -f dockerfiles/train_model.dockerfile .
```

In local environment, you can uncomment the following lines in `train_model.dockerfile` to enable authentication with Google Cloud:
```sh
# COPY ./service_account.json /root/.config/gcloud/application_default_credentials.json
# ENV GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
```
and make sure that you have a JSON key file for a service account with access to the project in the root directory of the project.

To run the docker image, run:
```sh
docker run trainer:latest
```

optionaly, you can specify some arguments:
```sh
  docker run trainer:latest \
    experiment=experiment[X].yaml \
    deployment.wandb.api_key=[FILL_IN_YOUR_WANDB_API_KEY]
```
or any other Hydra overrides.

### Training
To run a job on Vertex AI, after being logged in with Service Account, run:
```sh
gcloud ai custom-jobs create --region=europe-west1 --display-name=team94-train --config=train_vertex.json
```

**train_vertex.json**
```json
{
  "workerPoolSpecs": [
    {
      "machineSpec": {
        "machineType": "n1-highcpu-16"
      },
      "replicaCount": "1",
      "diskSpec": {
        "bootDiskType": "pd-ssd",
        "bootDiskSizeGb": 100
      },
      "containerSpec": {
        "imageUri": "gcr.io/dtu-mlops-jh/trainer@sha256:478496c15d659109a28967b7a7e20918887d660c8215db4eb2022bd66bfbf189",
        "args": [
          "experiment=experiment0.yaml",
          "deployment.wandb.api_key=[FILL_IN_YOUR_WANDB_API_KEY]",
          "experiment.dataset.path=/gcs/mlops_group94/data/processed/seabed_dataset/"
        ]
      }
    }
  ],
  "serviceAccount": "team94@dtu-mlops-jh.iam.gserviceaccount.com",
  "baseOutputDirectory": {
    "outputUriPrefix": "gs://mlops_group94/team94_vertex_trainer/"
  }
}
```


### Run FastAPI server
```sh
  uvicorn --reload --port 8000 dtu_mlops_team94.app:app
```

Docker
```sh
docker build -t deploy-app:latest -f dockerfiles/deploy_model_app.dockerfile .
docker run -e PORT=8080 -p 8080:8080 deploy-app:latest
```
