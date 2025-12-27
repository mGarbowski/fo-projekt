"""Data loader to use for training

Include both raw and processed data.
"""
from typing import Any
import pickle

import numpy as np
import pandas as pd


DatasetItem = dict[str, Any]

# TODO include new features
# TODO drop unnecessary features
# TODO normalize
# TODO handle missing data
class DatasetProcessor:
    """Process the raw csv files into a format suitable for loading into a torch Dataset.

    Groups data by object id, groups sequence entries by passband, sorted by time.
    """

    def __init__(self, metadata_path: str, lightcurves_path: str, output_path: str):
        self.metadata_path = metadata_path
        self.lightcurves_path = lightcurves_path
        self.output_path = output_path

    def process(self) -> list[DatasetItem]:
        """Process the raw data into """
        self._load_raw_data()
        self._extract_labels_from_metadata()

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

        return {
            "object_id": object_id,
            "label": self._get_label(object_id),
            "metadata": self._get_metadata(object_id),
            "sequences": {
                passband: self._get_sequence(object_id, passband)
                for passband in range(6)
            }
        }