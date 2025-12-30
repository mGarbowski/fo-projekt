"""Data loader to use for training

Include both raw and processed data.
"""

import pickle
from pathlib import Path
from typing import Any, Literal, TypedDict, final

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from supernova.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SCALER_DIR

N_BANDS = 6

DatasetItem = dict[str, Any]


@final
class DatasetProcessor:
    """Process the raw csv files into a format suitable for loading into a torch Dataset.

    Groups data by object id, groups sequence entries by passband, sorted by time.
    """

    metadata_path: Path
    lightcurves_path: Path
    output_path: Path
    metadata: pd.DataFrame
    labels: pd.DataFrame
    lightcurves: pd.DataFrame
    overwrite: bool = True

    def __init__(
        self,
        metadata_path: Path,
        lightcurves_path: Path,
        output_path: Path,
        overwrite: bool = True,
    ):
        self.overwrite = overwrite
        self.metadata_path = metadata_path
        self.lightcurves_path = lightcurves_path
        self.output_path = output_path

    def process(self) -> list[DatasetItem]:
        if self.output_path.exists():
            if self.overwrite:
                print("Overwriting existing processed dataset file.")
            else:
                print("Processed dataset file already exists on disk. Loading it.")
                return self.load_from_file(self.output_path)
        else:
            print("Processed dataset not found on disk, processing...")

        self._load_raw_data()
        self._extract_labels_from_metadata()
        self._remap_labels()
        self._drop_unnecessary_features()
        self._fix_missing_data()
        self._add_time_series_features()
        self._add_metadata_features()
        self._normalize_time_series()
        self._normalize_metadata()
        self._convert_to_float()

        return [
            self._get_single_item(obj_id)
            for obj_id in tqdm(
                self._get_all_object_ids(), "Saving processed dataset to file..."
            )
        ]

    def save_to_file(self, data: list[DatasetItem]):
        """Save processed data to output file."""

        with open(self.output_path, "wb") as out_file:
            pickle.dump(data, out_file)

    @staticmethod
    def load_from_file(path: Path) -> list[DatasetItem]:
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
        seq = seq[
            [
                "delta_t",
                "delta_t_cumsum",
                "flux_norm",
                "flux_err_norm",
                "snr",
                "detected",
            ]
        ]
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

    def _drop_unnecessary_features(self):
        self.metadata.drop(
            columns=[
                "ra",
                "decl",
                "gal_l",
                "gal_b",
                "hostgal_photoz",
                "hostgal_photoz_err",
            ],
            inplace=True,
        )

    def _fix_missing_data(self):
        """Flag extragalactic objects and fill missing distance modulus values with 0."""
        self.metadata["is_extragalactic"] = self.metadata["distmod"].isnull()
        self.metadata["distmod"] = self.metadata["distmod"].fillna(0)

    def _add_time_series_features(self):
        """Add features:
        * delta_t (scaled to account for time dilation)
        * cumulative delta_t
        * signal to noise ratio
        """
        # add delta_t
        seqs = self.lightcurves
        seqs.sort_values(["object_id", "passband", "mjd"], inplace=True)
        seqs["delta_t"] = (
            seqs.groupby(["object_id", "passband"])["mjd"].diff().fillna(0.0)
        )
        # scale delta_t by (1 + z)
        seqs_merged = seqs.merge(
            self.metadata[["hostgal_specz"]],
            left_on="object_id",
            right_index=True,
            how="left",
        )
        seqs["delta_t"] /= 1.0 + seqs_merged["hostgal_specz"]

        # delta_t_cumsum
        seqs["delta_t_cumsum"] = seqs.groupby(["object_id", "passband"])[
            "delta_t"
        ].cumsum()

        # snr
        seqs["snr"] = seqs["flux"].abs() / seqs["flux_err"]
        self.lightcurves = seqs

    def _add_metadata_features(self):
        """Add metadata features:
        * number of observations
        * number of detections
        * span of observations
        * max snr per band
        * mean flux per band"""
        meta, seqs = self.metadata, self.lightcurves
        n_obs = seqs.groupby("object_id").size().rename("n_obs")
        meta = meta.merge(n_obs, on="object_id", how="left")

        n_detected = seqs.groupby("object_id")["detected"].sum().rename("n_detections")
        meta = meta.merge(n_detected, on="object_id", how="left")

        t_span = (
            seqs.groupby("object_id")["mjd"]
            .agg(lambda vals: vals.max() - vals.min())
            .rename("t_span")
        )
        meta = meta.merge(t_span, on="object_id", how="left")

        max_snr_wide = (
            seqs.groupby(["object_id", "passband"])["snr"]
            .max()
            .unstack("passband")
            .add_prefix("max_snr_")
            .fillna(0.0)
            .reset_index()
        )
        meta = meta.merge(max_snr_wide, how="left", on="object_id")

        mean_flux_wide = (
            seqs.groupby(["object_id", "passband"])["flux"]
            .mean()
            .unstack("passband")
            .add_prefix("mean_flux_")
            .fillna(0.0)
            .reset_index()
        )
        meta = meta.merge(mean_flux_wide, how="left", on="object_id")
        self.metadata = meta.set_index("object_id")

    def _normalize_time_series(self):
        seqs = self.lightcurves
        scale = (
            seqs.assign(abs_flux=seqs["flux"].abs())
            .groupby(["object_id", "passband"])["abs_flux"]
            .mean()
            .rename("flux_scale")
            .replace(0.0, 1.0)
        )

        seqs = seqs.merge(scale, on=["object_id", "passband"], how="left")
        seqs["flux_norm"] = seqs["flux"] / seqs["flux_scale"]
        seqs["flux_err_norm"] = seqs["flux_err"] / seqs["flux_scale"]
        seqs = seqs.drop(columns=["flux_scale"])

        # drop denormalized columns
        self.lightcurves = seqs.drop(columns=["flux", "flux_err"])

    def _normalize_metadata(self):
        """Normalize metadata and dump scalers to files"""
        meta = self.metadata

        # standard scaling
        for col in [
            "hostgal_specz",
            "distmod",
            *(f"max_snr_{i}" for i in range(N_BANDS)),
        ]:
            scaler = StandardScaler()
            meta[col] = scaler.fit_transform(meta[[col]])
            with open(SCALER_DIR / col, "wb") as f:
                pickle.dump(scaler, f)

        # log1p, then standardscaler
        mwebv_scaler = Pipeline(
            [
                ("log1p", FunctionTransformer(np.log1p, validate=False)),
                ("scaler", StandardScaler()),
            ]
        )
        meta["mwebv"] = mwebv_scaler.fit_transform(meta[["mwebv"]])
        with open(SCALER_DIR / "mwebv", "wb") as f:
            pickle.dump(mwebv_scaler, f)
        self.metadata = meta

    def _convert_to_float(self):
        """All columns must be float64 for torch compatibility."""
        self.metadata = self.metadata.astype(np.float64)


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


if __name__ == "__main__":
    processor = DatasetProcessor(
        metadata_path=RAW_DATA_DIR / "training_set_metadata.csv",
        lightcurves_path=RAW_DATA_DIR / "training_set.csv",
        output_path=PROCESSED_DATA_DIR / "training_set.pkl",
    )
    dataset = processor.process()
    processor.save_to_file(dataset)
