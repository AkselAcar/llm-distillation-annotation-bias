import yaml
from typing import Iterable
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        return torch.optim.AdamW(parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
    

def plot_metric_over_epochs(metric_array, metric_name, alpha_values, filename):
    try:
        epochs = len(metric_array[0])
    except TypeError:
        epochs = len(metric_array)
    print('EPOCHS', epochs)

    plt.figure()
    for i, alpha in enumerate(alpha_values):
        plt.plot(np.arange(epochs), metric_array[i], label=rf'$\alpha=${alpha:.2f}')
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_metric_to_csv(metric_array, filename):
    df = pd.DataFrame(metric_array)
    df.to_csv(filename, index=False)