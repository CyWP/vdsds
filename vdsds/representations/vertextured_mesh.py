from __future__ import annotations
import torch
import nvdiffrast.torch as dr

from typing import Any, Dict, Optional
from torch import Tensor
from jaxtyping import Float, Int, Bool

from .mesh import Mesh
from ..utils.camera import Camera
from ..utils.img import Splimage


class VerTexturedMesh(Mesh):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        uv_co: Optional[Float[Tensor, "V 2"]] = None,
        texture: Optional[Splimage] = None,
        color: Optional[Float[Tensor, "V 3"]] = None,
        **kwargs,
    ):
        super().__init__(V, F.to(torch.int32))
        if color is None:
            self.color = texture.image_sample(
                torch.stack([uv_co[:, 1], 1 - uv_co[:, 0]], dim=1)
            )[:, :, :3].contiguous()
        else:
            self.color = color

    def _tensors(self) -> Dict[str, Tensor]:
        return {**super()._tensors(), "color": self.color}

    def _apply_tensors(self, tensor_dict: Dict[str, Tensor]):
        super()._apply_tensors(tensor_dict)
        self.color = tensor_dict["color"]

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "color": self.color}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerTexturedMesh:
        return cls(V=data["V"], F=data["F"], color=data["color"])

    def copy(self) -> VerTexturedMesh:
        return VerTexturedMesh(
            V=self.V.clone(),
            F=self.F.clone(),
            color=self.color.clone(),
        )

    def rasterize(
        self, camera: Camera, antialias: bool = True
    ) -> Float[Tensor, "B 4 H W"]:
        pos_clip = self.VH @ self.camera_to_nvdiffrast(camera).T
        rast, _ = dr.rasterize(self.ctx, pos_clip[None], self.F, [camera.H, camera.W])
        rgb, _ = dr.interpolate(
            self.color,
            rast,
            self.F,
        )
        normal_align = self.vertex_normal_alignment(self.up[None])[:, None] * 0.3 + 0.9
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
