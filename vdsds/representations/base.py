from __future__ import annotations

import torch

from jaxtyping import Float
from torch import Tensor
from typing import Dict, List, Type

from ..utils.camera import Camera


class Model:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def device(self) -> torch.device:
        for v in self._tensors().values():
            return v.device
        raise ValueError(f"{self.__class__.__name__} has no tensors")

    @property
    def dtype(self) -> torch.dtype:
        for v in self._tensors().values():
            if v.is_floating_point():
                return v.dtype
        for v in self._tensors().values():
            return v.dtype
        raise ValueError(f"{self.__class__.__name__} has no tensors")

    def _tensors(self) -> Dict[str, Tensor]:
        raise NotImplementedError

    def to(self, *args, **kwargs) -> Model:
        mapped = {}
        for k, v in self._tensors().items():
            mapped[k] = v.to(*args, **kwargs)
        self._apply_tensors(mapped)
        return self

    def _apply_tensors(self, tensor_dict: Dict[str, Tensor]):
        raise NotImplementedError

    def requires_grad_(self, mode: bool = True) -> Model:
        for v in self._tensors().values():
            v.requires_grad_(mode)
        return self

    def parameters(self) -> List[Tensor]:
        return [v for v in self._tensors().values() if v.is_floating_point()]

    def to_dict(self) -> Dict[str, Tensor]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict[str, Tensor]) -> Model:
        raise NotImplementedError

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
