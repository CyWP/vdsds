from __future__ import annotations

from typing import Any

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float, Int
from torch import Tensor

from .base import Model
from .mesh import Mesh


class TexturedMesh(Mesh):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        uv_co: Float[Tensor, "V 2"],
        texture: Float[Tensor, "B C H W"],
        **kwargs,
    ):
        super().__init__(V, F.to(torch.int32), texture=texture)
        # self.texture = texture.contiguous()
        self.uv_co = uv_co.contiguous()

    def _tensors(self) -> dict[str, Tensor]:
        return {**super()._tensors(), "uv_co": self.uv_co}

    def _apply_tensors(self, tensor_dict: dict[str, Tensor]):
        super()._apply_tensors(tensor_dict)
        self.uv_co = tensor_dict["uv_co"]

    def _dict_data(self) -> dict[str, Any]:
        return {**super()._dict_data(), "uv_co": self.uv_co}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TexturedMesh:
        return cls(
            V=data["V"],
            F=data["F"],
            uv_co=data["uv_co"],
            texture=data["texture"],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TexturedMesh:
        return Model.from_dict(data)

    def raster_albedo(self, rast) -> Float[Tensor, "B H W 4"]:
        uv_img, _ = dr.interpolate(self.uv_co[None], rast, self.F)
        return dr.texture(
            self.texture[:, :3].permute(0, 2, 3, 1).contiguous(),
            uv_img,
            filter_mode="linear",
            boundary_mode="wrap",
        )
