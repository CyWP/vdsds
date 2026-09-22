import torch

from ..utils.config import Config
from .base import Script
from .train_sds import TrainModelSDS
from .view_model import ViewModel


def get_script(name: str, device: torch.device, config: Config, **kwargs) -> Script:
    if name == "view":
        return ViewModel(device, config, **kwargs)
    elif name == "train_sds":
        return TrainModelSDS(device, config, **kwargs)
    else:
        raise ValueError(f"Task '{name}' is unrecognized.")


def run_script(name: str, device: torch.device, config: Config, **kwargs) -> None:
    get_script(name, device, config, **kwargs).launch()
