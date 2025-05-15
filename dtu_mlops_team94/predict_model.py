import torch
from dtu_mlops_team94.data.seabed_dataset import SeabedDataset
from dtu_mlops_team94.data.samplers_factory import SamplersFactory
from dtu_mlops_team94.models.model import SeabedClassifier
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import dtu_mlops_team94.utils as utils
import wandb
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns:
        Predictions array
    """
    # Run inference
    predictions = []
    accuracy = 0
    
    for batch in dataloader:
        x, y = batch
        y_pred = model(x)

        y_pred_argmax = np.argmax(y_pred.detach().numpy())
        predictions.append(y_pred_argmax)

    accuracy = np.sum(np.array(predictions) == y) / len(y)

    return predictions, accuracy


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def do_inference(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))

    # Set the environment variables (W&B API key)
    utils.env.set_environment(config)

    # Set the seed for reproducible results
    utils.env.set_seed(config.experiment.seed)

    # Load the dataset
    seabed_dataset = SeabedDataset(config.experiment.dataset.path)

    # Train - val - test split
    _, _, test_sampler = SamplersFactory.create(seabed_dataset)

    # Create the dataloader
    test_dataloader = torch.utils.data.DataLoader(seabed_dataset, batch_size=1, shuffle=False, sampler=test_sampler)

    model_path = ""
    # Download the model from W&B
    # with wandb.init(project=config.project.wandb.project_name, job_type="inference") as run:
    #     artifact = run.use_artifact("jakubh/model-registry/SeabedClassifier:latest", type="model")
    #     artifact_dir = artifact.download()

    #     logger.info(f"Artifact downloaded to: {artifact_dir}")

    #     # find the name of the .pth file
    #     for file in os.listdir(artifact_dir):
    #         if file.endswith(".pth"):
    #             model_path = os.path.join(artifact_dir, file)
    #             break

    model_path = os.path.join(hydra.utils.get_original_cwd(), "models/model_20240118-103658.ckpt")
    # Load the model
    model = SeabedClassifier.load_from_checkpoint(model_path)
    # model = torch.load(model_path)

    # model.eval()

    # Run prediction
    y_pred, accuracy = predict(model, test_dataloader)

    # Log the accuracy
    logger.info(f"Accuracy: {accuracy}")

    # Print the predictions
    logger.info(y_pred)


if __name__ == "__main__":
    do_inference()
