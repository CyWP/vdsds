from __future__ import annotations

from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from .conventions import UP


class LightSource:
    """
    Just a representation of a point light
    """

    _up: ClassVar = [UP[0], UP[2], UP[1]]

    def __init__(
        self,
        origin: Float[Tensor, 3] | None = None,
        strength: Float[Tensor, ""] | None = None,
        ambient: Float[Tensor, ""] | None = None,
    ):
        self.origin = torch.tensor(self._up) if origin is None else origin
        self.strength = torch.tensor(1.5) if strength is None else strength

    def device(self) -> torch.device:
        return self.origin.device

    @property
    def dtype(self) -> torch.dtype:
        return self.origin.dtype

    def to(self, *args, **kwargs) -> LightSource:
        self.origin = self.origin.to(*args, **kwargs)
        self.strength = self.strength.to(*args, **kwargs)
        return self

    def requires_grad_(self, mode: bool) -> LightSource:
        self.origin.requires_grad_(mode)
        self.strength.requires_grad_(mode)
        return self

    def copy(self) -> LightSource:
        """
        Return a copy of the object.
        """
        return LightSource(self.origin.clone(), self.strength.clone())
