from __future__ import annotations

import torch

from typing import Dict, Type
from torch import Tensor, nn
from jaxtyping import Float

from ..utils.camera import Camera
from ..rasterizable import Rasterizable
from ..representations.base import Model


class Deformation(nn.Module, Rasterizable):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
        next(self.parameters()).dtype

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> Deformation:
        model_keys = {}
        direct_keys = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_keys[key[6:]] = value
            else:
                direct_keys[key] = value

        model_cls = cls._get_model_type()
        model = model_cls.from_state_dict(model_keys)
        return cls(model=model, **direct_keys)

    @classmethod
    def _get_model_type(cls) -> Type[Model]:
        hints = cls.__init__.__annotations__
        if "model" in hints:
            return hints["model"]
        raise ValueError(f"Cannot determine model type for {cls.__name__}")

    def __len__(self) -> int:
        return len(self.model)

    def deformed(self, camera: Camera) -> Model:
        raise NotImplementedError()

    def rasterize(self, camera: Camera) -> Float[Tensor, "B 4 H W"]:
        return self.deformed(camera).rasterize(camera)

    def train(self, train_model: bool = False):
        self.requires_grad_(True)
        self.model.requires_grad_(False)
