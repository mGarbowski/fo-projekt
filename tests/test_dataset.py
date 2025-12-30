import torch

from supernova.dataset import get_dataset_split
from supernova.config import N_BANDS, PROCESSED_TRAINING_SET_FILE


def test_dataset_split_is_deterministic():
    val_split = 0.15
    test_split = 0.15

    split_1 = get_dataset_split(PROCESSED_TRAINING_SET_FILE, val_split, test_split)
    split_2 = get_dataset_split(PROCESSED_TRAINING_SET_FILE, val_split, test_split)

    for split in ["train", "val", "test"]:
        item_1 = split_1[split][0]
        item_2 = split_2[split][0]

        assert item_1["object_id"] == item_2["object_id"]
        assert item_1["label"] == item_2["label"]
        assert torch.equal(item_1["metadata"], item_2["metadata"])
        for band in range(N_BANDS):
            assert torch.equal(item_1["sequences"][band], item_2["sequences"][band])
            assert torch.equal(item_1["lengths"][band], item_2["lengths"][band])
