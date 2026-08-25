from __future__ import annotations

import torch
import torch.nn as nn

from jaxtyping import Float
from torch import Tensor
from typing import List

from ..utils.camera import Camera


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def register_parameter(self, name: str, value: Tensor | nn.Parameter) -> None:
        if isinstance(value, nn.Parameter):
            setattr(self, name, value)
        else:
            super().register_parameter(name, nn.Parameter(value))

    def combine(self, models: List[Model]) -> Model:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def centroid(self) -> torch.Tensor:
        raise NotImplementedError()

    def rasterize(self, camera: Camera) -> Float[Tensor, "B 4 H W"]:
        raise NotImplementedError()

    def forward(self, camera: Camera) -> Float[Tensor, "B 4 H W"]:
        return self.rasterize(camera)
