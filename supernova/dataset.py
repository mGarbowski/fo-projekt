"""Data loader to use for training

Include both raw and processed data.
"""

import pickle
from typing import Any, TypedDict, final

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

N_BANDS = 6

DatasetItem = dict[str, Any]


# TODO include new features
# TODO drop unnecessary features
# TODO normalize
# TODO handle missing data
@final
class DatasetProcessor:
    """Process the raw csv files into a format suitable for loading into a torch Dataset.

    Groups data by object id, groups sequence entries by passband, sorted by time.
    """

    metadata_path: str
    lightcurves_path: str
    output_path: str
    metadata: pd.DataFrame
    labels: pd.DataFrame
    lightcurves: pd.DataFrame

    def __init__(self, metadata_path: str, lightcurves_path: str, output_path: str):
        self.metadata_path = metadata_path
        self.lightcurves_path = lightcurves_path
        self.output_path = output_path

    def process(self) -> list[DatasetItem]:
        """Process the raw data into"""
        self._load_raw_data()
        self._extract_labels_from_metadata()
        self._remap_labels()

        return [self._get_single_item(obj_id) for obj_id in self._get_all_object_ids()]

    def save_to_file(self, data: list[DatasetItem]):
        """Save processed data to output file."""

        with open(self.output_path, "wb") as out_file:
            pickle.dump(data, out_file)

    @staticmethod
    def load_from_file(path: str) -> list[DatasetItem]:
        """Load processed data from output file."""

        with open(path, "rb") as in_file:
            return pickle.load(in_file)

    def _load_raw_data(self):
        """Load raw data from csv files."""

        self.metadata = pd.read_csv(self.metadata_path).set_index("object_id")
        self.lightcurves = pd.read_csv(self.lightcurves_path)

    def _extract_labels_from_metadata(self):
        """Extract dataframe mapping object id to class labels."""

        self.labels = self.metadata[["target"]]
        self.metadata = self.metadata.drop(columns=["target"])

    def _remap_labels(self):
        """Original labels are random ints, remap to ints from range [0, num_classes)."""

        unique_labels = self.labels["target"].unique()
        unique_labels.sort()
        label_mapping = {
            old_label: new_label for new_label, old_label in enumerate(unique_labels)
        }
        self.labels["target"] = self.labels["target"].map(label_mapping)

    def _get_all_object_ids(self) -> np.ndarray:
        """Get all unique object ids from the lightcurves data."""

        return self.lightcurves["object_id"].unique()

    def _get_sequence(self, object_id: int, passband: int) -> np.ndarray:
        """Get the sequence of observations for a given object id and passband."""

        lightcurves = self.lightcurves
        seq = lightcurves[
            (lightcurves["object_id"] == object_id)
            & (lightcurves["passband"] == passband)
        ].copy()
        seq.sort_values(by="mjd", inplace=True)
        seq = seq[["mjd", "flux", "flux_err", "detected"]]
        return seq.values

    def _get_metadata(self, object_id: int) -> np.ndarray:
        """Get the metadata for a given object id."""

        return self.metadata.loc[object_id].values

    def _get_label(self, object_id: int) -> np.int64:
        """Get the label for a given object id."""

        return self.labels.loc[object_id].iloc[0]

    def _get_single_item(self, object_id: int) -> DatasetItem:
        """Create a single dataset item for a given object id."""

        sequences = {}
        lengths = {}
        for passband in range(N_BANDS):
            seq = self._get_sequence(object_id, passband)
            sequences[passband] = seq
            lengths[passband] = len(seq)

        return {
            "object_id": object_id,
            "label": self._get_label(object_id),
            "metadata": self._get_metadata(object_id),
            "sequences": sequences,
            "lengths": lengths,
        }


class SupernovaDatasetEntry(TypedDict):
    object_id: int
    label: torch.Tensor
    metadata: torch.Tensor
    sequences: dict[int, torch.Tensor]
    lengths: dict[int, torch.Tensor]


@final
class SupernovaDataset(Dataset[SupernovaDatasetEntry]):
    """Torch Dataset for Supernova data."""

    def __init__(self, dataset_path: str):
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
        for band_id in range(6)
    }

    band_sequences = {
        band_id: pad_sequence(
            [item["sequences"][band_id] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        for band_id in range(6)
    }

    return {
        "metadata": metadata,
        "sequences": band_sequences,
        "lengths": lengths,
        "labels": labels,
    }
