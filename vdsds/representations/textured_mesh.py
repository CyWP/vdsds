from __future__ import annotations

from typing import Any

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float, Int
from torch import Tensor

from ..utils.camera import Camera
from ..utils.img import Splimage
from .mesh import Mesh


class TexturedMesh(Mesh):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        uv_co: Float[Tensor, "V 2"],
        texture: Splimage,
    ):
        super().__init__(V, F.to(torch.int32))
        self.texture = Splimage(texture._tensor.contiguous())
        self.uv_co = uv_co.contiguous()

    def _tensors(self) -> dict[str, Tensor]:
        return {**super()._tensor(), "texture": self.texture, "uv_co": self.uv_co}

    def _apply_tensors(self, tensor_dict: dict[str, Tensor]):
        super()._apply_tensors(tensor_dict)
        self.uv_co = tensor_dict["uv_co"]
        self.texture._tensor = tensor_dict["texture"]

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "uv_co": self.uv_co,
            "texture": self.texture,
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

    def rasterize(
        self, camera: Camera, antialias: bool = True
    ) -> Float[Tensor, "B 4 H W"]:
        pos_clip = self.VH @ self.camera_to_nvdiffrast(camera).T
        rast, _ = dr.rasterize(self.ctx, pos_clip[None], self.F, [camera.H, camera.W])
        uv_img, _ = dr.interpolate(self.uv_co[None], rast, self.F)
        rgb = dr.texture(
            self.texture._tensor[:, :3].permute(0, 2, 3, 1).contiguous(),
            uv_img,
            filter_mode="linear",
            boundary_mode="wrap",
        )
        normal_align = self.vertex_normal_alignment(self.up[None]) * 0.3 + 0.9
        shading, _ = dr.interpolate(normal_align, rast, self.F)
        rgba = torch.cat(
            [
                (rgb * shading).clamp(0, 1),
                torch.ones((*rgb.shape[:3], 1), device=rgb.device),
            ],
            dim=-1,
        )
        rgba = torch.where(rast[..., 3:4] > 0, rgba, torch.zeros_like(rgba))
        if antialias:
            pos_clip_batched = pos_clip[None] if pos_clip.ndim == 2 else pos_clip
            rgba = dr.antialias(rgba, rast, pos_clip_batched, self.F)
        return rgba.permute(0, 3, 1, 2)
