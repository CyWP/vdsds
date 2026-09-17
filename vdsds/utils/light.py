from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


class LightSource:
    """
    Just a representation of a point light
    """

    def __init__(
        self,
        origin: Float[Tensor, 3],
        strength: Float[Tensor, ""],
    ):
        self.origin = origin
        self.strength = strength

    @property
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
        self.Q.requires_grad_(mode)
        return self

    def copy(self) -> LightSource:
        """
        Return a copy of the object.
        """
        return LightSource(self.origin.clone(), self.strength.clone())
