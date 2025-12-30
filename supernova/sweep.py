import os
from typing import TypedDict, cast

import yaml
from lightning.pytorch.loggers import WandbLogger

import wandb
from supernova.config import (
    SWEEP_CONFIG_FILE,
    PROCESSED_TRAINING_SET_FILE,
    METADATA_INPUT_SIZE,
    LIGHTCURVE_INPUT_SIZE,
    NUM_CLASSES,
    VAL_SPLIT,
    TEST_SPLIT,
)
from supernova.dataset import get_data_loaders, get_dataset_split
from supernova.modeling.model import SupernovaClassifierV1Config
from supernova.modeling.train import SupernovaTraining, get_trainer


PROJECT_NAME = "supernova"


class SweepConfig(TypedDict):
    learning_rate: float

    metadata_num_hidden_layers: int
    metadata_hidden_size: int
    metadata_output_size: int

    lightcurve_num_hidden_layers: int
    lightcurve_hidden_size: int

    classifier_hidden_size: int
    classifier_num_hidden_layers: int

    dropout: float

    epochs: int
    early_stop_patience: int
    batch_size: int


def runner():
    with wandb.init(project=PROJECT_NAME) as run:
        config = cast(SweepConfig, run.config)

        model_config = SupernovaClassifierV1Config(
            metadata_input_size=METADATA_INPUT_SIZE,
            metadata_num_hidden_layers=config["metadata_num_hidden_layers"],
            metadata_hidden_size=config["metadata_hidden_size"],
            metadata_output_size=config["metadata_output_size"],
            lightcurve_input_size=LIGHTCURVE_INPUT_SIZE,
            lightcurve_num_hidden_layers=config["lightcurve_num_hidden_layers"],
            lightcurve_hidden_size=config["lightcurve_hidden_size"],
            classifier_hidden_size=config["classifier_hidden_size"],
            classifier_num_hidden_layers=config["classifier_num_hidden_layers"],
            num_classes=NUM_CLASSES,
            dropout=config["dropout"],
        )

        training_module = SupernovaTraining(
            model_config=model_config, learning_rate=config["learning_rate"]
        )

        checkpoint_dir = os.path.join(run.dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        trainer = get_trainer(
            epochs=config["epochs"],
            checkpoint_dir=checkpoint_dir,
            early_stop_patience=config["early_stop_patience"],
            logger=WandbLogger(project=PROJECT_NAME),
        )

        dataset_split = get_dataset_split(
            PROCESSED_TRAINING_SET_FILE, VAL_SPLIT, TEST_SPLIT
        )
        loaders = get_data_loaders(dataset_split, config["batch_size"])
        train_loader, val_loader = loaders["train"], loaders["val"]

        trainer.fit(
            training_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


if __name__ == "__main__":
    with open(SWEEP_CONFIG_FILE, "r") as f:
        cfg = cast(SweepConfig, yaml.safe_load(f))

    sweep_id = wandb.sweep(
        sweep=cfg,  # pyright: ignore[reportArgumentType]
        project=PROJECT_NAME,
    )
    wandb.agent(sweep_id, count=512, function=runner)
