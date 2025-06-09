'''
The functions in this file were previously in utils.py, but due to
circular imports with model.py, it was necessary to move them
'''

from torch.utils.data import DataLoader
from tqdm import tqdm
import constants
from model import HateSpeechDataset
from pathlib import Path
import os
import pandas as pd
import numpy as np


def parse_datasets(cfg: dict, data_path: Path, batch_size: int) -> tuple[DataLoader, DataLoader, dict]:
    """
    Parse the datasets from the configuration file.

    Args:
        cfg (dict): Configuration dictionary containing the dataset info.
        data_path (Path): Path to the data.
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loaders_dict)
    """
    num_datasets = len(cfg)
    if num_datasets == 0:
        raise ValueError("No datasets provided in the configuration file.")

    loader_train = None
    loader_val = None
    loader_tests = {}

    for dataset in tqdm(cfg, desc="Loading datasets", unit="dataset"):
        set_type = dataset.get("set", None)
        if set_type == "train":
            loader_train = load_data(cfg=dataset, data_path=data_path, batch_size=batch_size)
        elif set_type == "val":
            if dataset.get("name", None) is not None:
                loader_val = load_data(cfg=dataset, data_path=data_path, batch_size=batch_size)
            else:
                loader_val = None
        elif set_type.startswith("test"):
            loader_tests[set_type] = load_data(cfg=dataset, data_path=data_path, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown dataset type: {set_type}")

    return loader_train, loader_val, loader_tests


def load_data(cfg: dict, data_path: Path, batch_size: int) -> DataLoader:
    """Load the data from the path and return a DataLoader.

    Args:
        cfg (dict): Configuration dictionary containing the informations about the data

    Returns:
        loader (DataLoader): A DataLoader object containing the data.
    """
    # Go to second-level dataset directory
    file_path = data_path / "second-level"

    dataset_file = os.path.join(file_path, cfg.get("name", None))
    df = pd.read_csv(dataset_file)

    split = cfg.get("split", None)
    if split is not None:
        start = int(split[0] * len(df))
        end = int(split[1] * len(df))
        df = df.iloc[start:end]

    # Convert columns to Numpy arrays to use them as input of model generation
    texts = df['text'].to_numpy()
    hard_labels = df['y'].to_numpy()
    soft_targets_yes = df['soft_yes'].to_numpy()
    soft_targets_no = df['soft_no'].to_numpy()

    # Shape when transposed : (N, 2) with N the number of entries in the dataset
    soft_targets = np.array([soft_targets_yes, soft_targets_no]).T
    """
    # Get split to apply to the dataset (if any)
    split = cfg.get("split", None)

    if split is not None:
        start = int(split[0] * len(df))
        end = int(split[1] * len(df))
        df = df.iloc[start:end]
    """
    # Get additional parameters
    shuffle = cfg.get("shuffle", True)

    # Create dataset
    dataset = HateSpeechDataset(texts, hard_labels, soft_targets)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
