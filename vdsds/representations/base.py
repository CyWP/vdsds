from __future__ import annotations

import torch
import torch.nn as nn

from jaxtyping import Float
from torch import Tensor
from typing import Dict, List, Type

from ..utils.camera import Camera


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
        next(self.parameters()).dtype

    def register_parameter(self, name: str, value: Tensor | nn.Parameter) -> None:
        if isinstance(value, nn.Parameter):
            super().register_parameter(name, value)
        else:
            super().register_parameter(name, nn.Parameter(value))

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> Model:
        direct_params = {}
        submodule_dicts: Dict[str, Dict[str, Tensor]] = {}

        for key, value in state_dict.items():
            if "." in key:
                prefix, rest = key.split(".", 1)
                submodule_dicts.setdefault(prefix, {})[rest] = value
            else:
                direct_params[key] = value

        submodules = {}
        for name, sub_dict in submodule_dicts.items():
            sub_cls = cls._get_submodule_type(name)
            submodules[name] = sub_cls.from_state_dict(sub_dict)

        return cls(**direct_params, **submodules)

    @classmethod
    def _get_submodule_type(cls, name: str) -> Type[Model]:
        hints = cls.__init__.__annotations__
        if name in hints:
            return hints[name]
        raise ValueError(f"Cannot determine type for submodule '{name}' in {cls.__name__}")

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
