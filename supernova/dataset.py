"""PyTorch Dataset and DataLoader for Supernova data.

Loaded for preprocessed .pkl file
"""

from pathlib import Path
from typing import Literal, TypedDict, final

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from supernova.config import N_BANDS
from supernova.preprocessing import DatasetProcessor


class SupernovaDatasetEntry(TypedDict):
    object_id: int
    label: torch.Tensor
    metadata: torch.Tensor
    sequences: dict[int, torch.Tensor]
    lengths: dict[int, torch.Tensor]


@final
class SupernovaDataset(Dataset[SupernovaDatasetEntry]):
    """Torch Dataset for Supernova data."""

    def __init__(self, dataset_path: Path):
        self._data = DatasetProcessor.load_from_file(dataset_path)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        item = self._data[idx]

        sequences = {
            band_id: torch.tensor(seq, dtype=torch.float32)
            for band_id, seq in item["sequences"].items()
        }

        lengths = {
            band_id: torch.tensor(length, dtype=torch.long)
            for band_id, length in item["lengths"].items()
        }

        return SupernovaDatasetEntry(
            {
                "object_id": int(item["object_id"]),
                "label": torch.tensor(item["label"], dtype=torch.long),
                "metadata": torch.tensor(item["metadata"], dtype=torch.float32),
                "sequences": sequences,
                "lengths": lengths,
            }
        )


def supernova_collate_fn(batch):
    """Collate function for SupernovaDataset to prepare data for SupernovaClassifierV1."""
    metadata = torch.stack([item["metadata"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    lengths = {
        band_id: torch.stack([item["lengths"][band_id] for item in batch])
        for band_id in range(N_BANDS)
    }

    band_sequences = {
        band_id: pad_sequence(
            [item["sequences"][band_id] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        for band_id in range(N_BANDS)
    }

    return {
        "metadata": metadata,
        "sequences": band_sequences,
        "lengths": lengths,
        "labels": labels,
    }


SplitName = Literal["train", "val", "test"]
SupernovaDatasetSplit = dict[SplitName, Subset[SupernovaDatasetEntry]]


def get_dataset_split(
    dataset_path: Path,
    val_split: float,
    test_split: float,
    random_seed: int = 42,
) -> SupernovaDatasetSplit:
    assert 0 < val_split < 1
    assert 0 < test_split < 1
    assert val_split + test_split < 1

    dataset = SupernovaDataset(dataset_path)

    total_size = len(SupernovaDataset(dataset_path))
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_data_loaders(
    datasets: SupernovaDatasetSplit,
    batch_size: int,
    num_workers: int = 4,
) -> dict[SplitName, DataLoader[SupernovaDatasetEntry]]:
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=supernova_collate_fn,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=supernova_collate_fn,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=supernova_collate_fn,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
