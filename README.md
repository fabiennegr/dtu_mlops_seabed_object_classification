# MLOps project - Seabed Object Classification

A project from [Machine Learning Operations (MLOps)](https://skaftenicki.github.io/dtu_mlops/) course at DTU (EuroTeQ).

## Goal

This is the project description for group 94 in the 02476 Machine Learning Operations course at DTU. The overall goal of the project is to apply the material we have learned in the course to a machine learning problem, while using one of the given frameworks to do so.

## Framework and Deep Learning Model

In our project, we harnessed the power of the PyTorch Framework, coupled with the PyTorch Lightning extension, to comprehensively develop, train, and evaluate our deep learning model. This dynamic combination facilitated a streamlined workflow, utilizing PyTorch's dynamic computation graph and user-friendly interface alongside the enhanced structure provided by PyTorch Lightning. In addition to torchvision, we integrated the ResNet model as a baseline, leveraging its robust features. The PyTorch Lightning extension further simplified the training process, enabling us to focus on model fine-tuning and optimization for our specific deep learning task. Together, these tools formed a cohesive and efficient framework for our project. We also used multiple "utility" libraries - such as numpy and pandas for data manipulation, hydra for configuration, wandb for tracking and storing our experiments, GCP SDK (google-cloud-secret-manager, etc.), followed by libraries like dvc, pytest, coverage, and fastapi with uvicorn for deploying our model.

## Data

We used DVC to store our compressed dataset in a ZIP archive. Our dataset is a set of [seabed images](https://universe.roboflow.com/hyit-mdvni/seabed) for a CV classification task, which means that there can be quite a large number of files, thus the data transfer might be slow - so we chose to compress the dataset into a ZIP archive and store it in the GCP Bucket using DVC. During the development, we also used DVC to store our trained models. Still, we switched to storing trained weights in Weights & Biases directly as Artifacts, as there is a clear connection between the model training and the weights, and it is more straightforward to use W&B for that. Storing the dataset and models using DVC helped us keep track of the versions, changes, and authors of the changes, and also easily share data between the team members.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── dtu_mlops_team94  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script, and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results-oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
