import torch
from data.seabed_dataset import SeabedDataset
from data.samplers_factory import SamplersFactory
from models.model import SeabedClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from datetime import datetime
from utils import env
import wandb

logger = logging.getLogger(__name__)


def train_dataloader(dataset, train_sampler):
    cores = env.get_cores_count()
    return torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=cores
    )


def val_dataloader(dataset, val_sampler):
    cores = env.get_cores_count()
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=cores)


def test_dataloader(dataset, test_sampler):
    cores = env.get_cores_count()
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=test_sampler, num_workers=cores)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def train(conf: DictConfig) -> None:
    config = conf.experiment
    cofig_proj = conf.project
    logger.info(OmegaConf.to_yaml(config))

    # Set the environment variables (W&B API key)
    env.set_environment(conf)

    # Set the seed for reproducible results
    env.set_seed(config.seed)

    # # Load file from GCP bucket
    # from google.cloud import storage

    # storage_client = storage.Client()
    # bucket = storage_client.bucket(config.bucket_name)
    # blob = bucket.blob(config.bucket_blob)
    # blob.download_to_filename(config.dataset.path)

    # Load the dataset
    seabed_dataset = SeabedDataset(config.dataset.path)

    # Preview a few images with their classes
    # preview_images(seabed_dataset, num_images=5)

    # Train - val - test split
    train_sampler, val_sampler, test_sampler = SamplersFactory.create(seabed_dataset)

    # Create the LightningTrainer and train the model
    trainer = pl.Trainer(
        max_epochs=config.hyperparameters.num_epochs,  # Number of training epochs
        logger=WandbLogger(
            name=config.wandb_name,
            project=cofig_proj.wandb.project_name,
            log_model=False,
        ),  # W&B integration
        callbacks=[
            EarlyStopping(
                monitor=config.early_stopping.monitor,
                mode=config.early_stopping.mode,
                patience=config.early_stopping.patience,
            ),
            # ModelCheckpoint(monitor="valid_acc", mode="max", filename=config.wandb_name),
        ],
    )

    model = SeabedClassifier(
        num_classes=seabed_dataset.num_classes,
        lr=config.hyperparameters.learning_rate,
        optimizer_name=config.hyperparameters.optimizer_name,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader(seabed_dataset, train_sampler),
        val_dataloaders=val_dataloader(seabed_dataset, val_sampler),
    )

    # Test the model
    # results = trainer.test(
    trainer.test(
        model,
        dataloaders=test_dataloader(seabed_dataset, test_sampler),
    )

    # Save the trained model
    timestring = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(hydra.utils.get_original_cwd(), f"models/model_{timestring}.ckpt")
    trainer.save_checkpoint(model_path)
    # torch.save(model.state_dict(), model_path)

    # @TODO: Save only the best model
    # https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#saving-loading-from-a-checkpoint

    # print("---- Test results ----")
    # print(results.test_acc)
    # print("----------------------")

    with wandb.init() as run:
        artifact = wandb.Artifact(f"{cofig_proj.wandb.project_name}-best", "model")

        artifact.add_file(model_path)
        run.log_artifact(artifact)
        run.link_artifact(artifact, "model-registry/SeabedClassifier")


if __name__ == "__main__":
    train()
