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


def parse_datasets(cfg: dict, data_path: Path, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Parse the datasets from the configuration file.

    Args:
        cfg (dict): Configuration dictionary containing the informations about the data

    Returns:
        tuple: A tuple containing the loaders for train, validation, and test datasets.
    """

    num_datasets = len(cfg)
    if num_datasets == 0:
        raise ValueError("No datasets provided in the configuration file.")

    for i in tqdm(range(num_datasets), desc="Loading datasets", unit="dataset"):
        dataset = cfg[i]
        set_type = dataset.get("set", None)
        if set_type == "train":
            loader_train = load_data(cfg=dataset, data_path=data_path, batch_size=batch_size)
        elif set_type == "val":
            if dataset.get("name", None) is not None :
                loader_val = load_data(cfg=dataset, data_path=data_path, batch_size=batch_size)
            else :
                loader_val = None
        elif set_type == "test":
            loader_test = load_data(cfg=dataset, data_path=data_path, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown dataset type: {set_type}")
    return loader_train, loader_val, loader_test

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

    # Convert columns to Numpy arrays to use them as input of model generation
    texts = df['text'].to_numpy()
    hard_labels = df['y'].to_numpy()
    soft_targets_yes = df['soft_yes'].to_numpy()
    soft_targets_no = df['soft_no'].to_numpy()

    # Shape when transposed : (N, 2) with N the number of entries in the dataset
    soft_targets = np.array([soft_targets_yes, soft_targets_no]).T

    # Get split to apply to the dataset (if any)
    split = cfg.get("split", None)

    if split is not None:
        start = int(split[0] * len(df))
        end = int(split[1] * len(df))
        df = df.iloc[start:end]

    # Get additional parameters
    shuffle = cfg.get("shuffle", True)

    # Create dataset
    dataset = HateSpeechDataset(texts, hard_labels, soft_targets)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
