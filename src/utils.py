import yaml
from typing import Iterable
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import constants
from model import HateSpeechDataset
from pathlib import Path
import os
import pandas as pd
import numpy as np


def parse_datasets(cfg: dict, data_path: Path) -> tuple[DataLoader, DataLoader, DataLoader]:
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
            loader_train = load_data(cfg=dataset, data_path=data_path)
        elif set_type == "val":
            loader_val = load_data(cfg=dataset, data_path=data_path)
        elif set_type == "test":
            loader_test = load_data(cfg=dataset, data_path=data_path)
        else:
            raise ValueError(f"Unknown dataset type: {set_type}")
    return loader_train, loader_val, loader_test

def load_data(cfg: dict, data_path: Path) -> DataLoader:
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
    batch_size = cfg.get("batch_size", constants.BATCH_SIZE)
    shuffle = cfg.get("shuffle", True)

    # Create dataset
    dataset = HateSpeechDataset(texts, hard_labels, soft_targets)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

def read_yml(cfg_file: str) -> dict:
    """Read a yaml configuration file.

    Args:
        cfg_file (str): Path to the yaml configuration file.

    Returns:
        dict : Configuration parameters as a dictionary.
    """
    with open(cfg_file, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


# Useful functions for the use of the .yml file
def get_criterion(criterion_name: str) -> nn.Module:
    """
    Get the criterion based on the name provided.

    Args:
        criterion_name (str): The name of the criterion.

    Returns:
        nn.Module: The criterion object.
    """
    if criterion_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {criterion_name} not recognized.")


def get_optimizer(
    optimizer_name: str, parameters: Iterable[Tensor], learning_rate: float
) -> Optimizer:
    """
    Get the optimizer based on the class name provided.

    Args:
        optimizer_name (str): The name of the optimizer class.
        parameters (iterable): The parameters to optimize.
        learning_rate (float): The learning rate.

    Returns:
        torch.optim.Optimizer: The optimizer object.
    """
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == "AdamW":
        return AdamW(parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")