from __future__ import annotations

from typing import Any

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float, Int
from torch import Tensor

from .mesh import Mesh


class VerTexturedMesh(Mesh):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        texture: Float[Tensor, "V 3"] | None = None,
        **kwargs,
    ):
        super().__init__(V, F.to(torch.int32), texture=texture)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerTexturedMesh:
        return cls(V=data["V"], F=data["F"], tetxure=data["texture"])

    def raster_albedo(self, rast) -> Float[Tensor, "B H W 4"]:
        return dr.interpolate(self.texture, rast, self.F)[0]
