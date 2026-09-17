from __future__ import annotations

from typing import Any

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float, Int
from torch import Tensor

from .mesh import Mesh


class TexturedMesh(Mesh):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        uv_co: Float[Tensor, "V 2"],
        texture: Float[Tensor, "B C H W"],
    ):
        super().__init__(V, F.to(torch.int32))
        self.texture = texture.contiguous()
        self.uv_co = uv_co.contiguous()

    def _tensors(self) -> dict[str, Tensor]:
        return {**super()._tensor(), "uv_co": self.uv_co}

    def _apply_tensors(self, tensor_dict: dict[str, Tensor]):
        super()._apply_tensors(tensor_dict)
        self.uv_co = tensor_dict["uv_co"]

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "uv_co": self.uv_co,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TexturedMesh:
        return cls(
            V=data["V"],
            F=data["F"],
            uv_co=data["uv_co"],
            texture=data["texture"],
        )

    def copy(self) -> TexturedMesh:
        return TexturedMesh(
            V=self.V.clone(),
            F=self.F.clone(),
            uv_co=self.uv_co.clone(),
            texture=self.texture.copy(),
        )

    def raster_albedo(self, rast) -> Float[Tensor, "B H W 4"]:
        uv_img, _ = dr.interpolate(self.uv_co[None], rast, self.F)
        return dr.texture(
            self.texture._tensor[:, :3].permute(0, 2, 3, 1).contiguous(),
            uv_img,
            filter_mode="linear",
            boundary_mode="wrap",
        )
