from . import gcp
from omegaconf import DictConfig
import os
import multiprocessing
import random
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_environment(config: DictConfig) -> None:
    """
    Sets the environment variables for the project.

    Args:
        config: Hydra config

    Returns:
        None
    """
    print("Setting environment variables")

    if config.deployment.wandb.api_key == "NONE":
        logger.info("No wandb key provided, getting it from Google Secret Manager")
        wandb_api_key = gcp.get_secret(config.project.gcp.project_id, config.project.gcp.wandb_api_key_secret_id)
    else:
        logger.info("Got wandb key from Hydra config")
        wandb_api_key = config.deployment.wandb.api_key

    # Set the W&B API key
    os.environ["WANDB_API_KEY"] = wandb_api_key


def set_seed(seed: int = 42) -> None:
    """Sets the seed for reproducible results.

    Args:
        seed: seed to use for reproducible results
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_cores_count():
    """
    Returns the number of cores avaiable (n_cores - 1)

    Returns:
        n_cores - 1
    """
    return multiprocessing.cpu_count() - 1
