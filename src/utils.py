import yaml
import torch
import torch.nn as nn
from transformers import AdamW


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
def get_criterion(criterion_name: str) -> torch.nn.Module:
    """
    Get the criterion based on the name provided.

    Args:
        criterion_name (str): The name of the criterion.

    Returns:
        torch.nn.Module: The criterion object.
    """
    if criterion_name == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    elif criterion_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
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