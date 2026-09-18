from __future__ import annotations

from typing import Any

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float, Int
from torch import Tensor

from ..utils.img import Splimage
from .mesh import Mesh


class VerTexturedMesh(Mesh):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        texture: Float[Tensor, "V 3"],
        **kwargs,
    ):
        super().__init__(V, F, texture=texture)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerTexturedMesh:
        return cls(V=data["V"], F=data["F"], tetxure=data["texture"])

    def raster_albedo(self, rast) -> Float[Tensor, "B H W 3"]:
        return dr.interpolate(self.texture, rast, self.F)[0]

    @classmethod
    def from_mesh_data(
        cls,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        uv_co: Float[Tensor, "V 2"],
        texture: Float[Tensor, "B C H W"],
        unit_box: bool = True,
        **kwargs,
    ) -> Mesh:
        if unit_box:
            V_min, V_max = V.min(dim=0).values, V.max(dim=0).values
            V_extent = V_max - V_min
            V_center = (V_max + V_min) / 2
            V = (V - V_center) / V_extent.max()
        V = torch.stack([V[:, 1], V[:, 0], -V[:, 2]], dim=1)
        texture = Splimage(texture, force_rgba=False).image_sample(uv_co)[0, :, :3]
        return cls(V, F, texture)
