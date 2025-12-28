"""Model trainer module"""

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.optim import Adam


# TODO parametrize more hyperparams
class SupernovaTraining(LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, metadata, sequences, lengths):
        return self.model(metadata, sequences, lengths)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def _step(self, batch, step_name: str):
        """Common step logic for training and validation steps."""
        metadata = batch["metadata"]
        sequences = batch["sequences"]
        lengths = batch["lengths"]
        labels = batch["labels"]

        logits = self(metadata, sequences, lengths)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(f"{step_name}_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)


def get_trainer(epochs: int, checkpoint_dir: str, early_stop_patience: int) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=checkpoint_dir,
        filename="supernova-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=early_stop_patience, mode="min"
    )

    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
    )

    return trainer
