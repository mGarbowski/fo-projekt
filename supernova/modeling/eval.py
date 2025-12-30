"""Model evaluation on test dataset."""

import json
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)
from tqdm import tqdm

from supernova.config import PROCESSED_TRAINING_SET_FILE, REVERSE_LABEL_MAPPING
from supernova.dataset import get_dataset_split, get_data_loaders
from supernova.modeling.model import SupernovaClassifierV1
from supernova.sweep import VAL_SPLIT, TEST_SPLIT


def load_model_from_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location=torch.device("cpu")
    )
    cfg = checkpoint["hyper_parameters"]["model_config"]
    model = SupernovaClassifierV1(cfg)

    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    return model


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass(frozen=True)
class EvaluationResults:
    predicted_labels: np.ndarray
    real_labels: np.ndarray
    confusion_matrix: np.ndarray
    report: str
    micro_averaged: Metrics
    macro_averaged: Metrics

    def display_report(self):
        print("=== Classification Report ===")
        print(self.report)
        print("=== Confusion Matrix ===")
        print(self.confusion_matrix)
        print("=== Micro-Averaged Metrics ===")
        print(f"Accuracy: {self.micro_averaged.accuracy:.2f}")
        print(f"Precision: {self.micro_averaged.precision:.2f}")
        print(f"Recall: {self.micro_averaged.recall:.2f}")
        print(f"F1 Score: {self.micro_averaged.f1_score:.2f}")
        print("=== Macro-Averaged Metrics ===")
        print(f"Accuracy: {self.macro_averaged.accuracy:.2f}")
        print(f"Precision: {self.macro_averaged.precision:.2f}")
        print(f"Recall: {self.macro_averaged.recall:.2f}")
        print(f"F1 Score: {self.macro_averaged.f1_score:.2f}")
        print("=" * 30)

    @classmethod
    def from_predictions(cls, predicted_labels: np.ndarray, real_labels: np.ndarray):
        cm = confusion_matrix(real_labels, predicted_labels)
        report = classification_report(
            real_labels,
            predicted_labels,
            target_names=[f"Klasa {lbl}" for lbl in REVERSE_LABEL_MAPPING],
        )

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            real_labels, predicted_labels, average="micro"
        )
        accuracy_micro = accuracy_score(real_labels, predicted_labels)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            real_labels, predicted_labels, average="macro"
        )
        accuracy_macro = (cm.diagonal() / cm.sum(axis=1)).mean()

        return cls(
            predicted_labels=predicted_labels,
            real_labels=real_labels,
            confusion_matrix=cm,
            report=report,
            micro_averaged=Metrics(
                accuracy=accuracy_micro,
                precision=precision_micro,
                recall=recall_micro,
                f1_score=f1_micro,
            ),
            macro_averaged=Metrics(
                accuracy=accuracy_macro,
                precision=precision_macro,
                recall=recall_macro,
                f1_score=f1_macro,
            ),
        )

    def to_dict(self):
        """Convert to JSON-serializable dictionary."""
        return {
            "predicted_labels": self.predicted_labels.tolist(),
            "real_labels": self.real_labels.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "report": self.report,
            "micro_averaged": asdict(self.micro_averaged),
            "macro_averaged": asdict(self.macro_averaged),
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct from dictionary."""
        return cls(
            predicted_labels=np.array(data["predicted_labels"]),
            real_labels=np.array(data["real_labels"]),
            confusion_matrix=np.array(data["confusion_matrix"]),
            report=data["report"],
            micro_averaged=Metrics(**data["micro_averaged"]),
            macro_averaged=Metrics(**data["macro_averaged"]),
        )

    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def evaluate_model(
    model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            metadata = batch["metadata"].to(device)
            sequences = {k: v.to(device) for k, v in batch["sequences"].items()}
            lengths = batch["lengths"]
            labels = batch["labels"]

            logits = model(metadata, sequences, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return EvaluationResults.from_predictions(
        predicted_labels=all_preds, real_labels=all_labels
    )


def main():
    parser = ArgumentParser(
        description="Evaluate a trained SupernovaClassifierV1 model on a test dataset."
    )
    parser.add_argument(
        "checkpoint_file", type=Path, help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        default=None,
        help="Optional path to save evaluation metrics as JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation (default: auto-detect)",
    )

    args = parser.parse_args()

    print(f"Loading model from checkpoint: {args.checkpoint_file}")
    model = load_model_from_checkpoint(args.checkpoint_file)
    print("Model loaded successfully.")

    print("Loading test dataset...")
    datasets = get_dataset_split(PROCESSED_TRAINING_SET_FILE, VAL_SPLIT, TEST_SPLIT)
    loaders = get_data_loaders(datasets, batch_size=32)
    test_loader = loaders["test"]
    print("Test dataset loaded.")

    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device=args.device)
    print("Evaluation complete.")

    results.display_report()

    if args.save_metrics:
        print(f"Saving evaluation metrics to: {args.save_metrics}")
        results.save(args.save_metrics)
        print("Metrics saved.")


if __name__ == "__main__":
    main()
