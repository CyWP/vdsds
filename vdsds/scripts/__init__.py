import torch

from ..utils.config import Config
from .analyze import AnalyzeDeformation
from .base import Script
from .orbit import OrbitFrames
from .train_sds import TrainModelSDS
from .view_model import ViewModel


def get_script(name: str, device: torch.device, config: Config, **kwargs) -> Script:
    if name == "view":
        return ViewModel(device, config, **kwargs)
    elif name == "train_sds":
        return TrainModelSDS(device, config, **kwargs)
    elif name == "orbit":
        return OrbitFrames(device, config, **kwargs)
    elif name == "analyze":
        return AnalyzeDeformation(device, config, **kwargs)
    else:
        raise ValueError(f"Task '{name}' is unrecognized.")


def run_script(name: str, device: torch.device, config: Config, **kwargs) -> None:
    get_script(name, device, config, **kwargs).launch()
