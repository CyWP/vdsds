from __future__ import annotations
import torch
import nvdiffrast.torch as dr

from typing import Any, Dict, Optional
from torch import Tensor
from jaxtyping import Float, Int, Bool

from .base import Model
from .mesh import Mesh
from .splat import Splat
from ..utils.quaternion import Quaternion
from ..utils.camera import Camera
from ..utils.img import Splimage


class TexturedMesh(Mesh):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Float[Tensor, "F 3"],
        uv_co: Float[Tensor, "V 2"],
        texture: Splimage,
    ):
        super().__init__(V, F.to(torch.int32))
        self.texture = Splimage(texture._tensor.contiguous())
        self.uv_co = uv_co.contiguous()
        self.opengl_conversion = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=V.device,
        )
        self.ctx = dr.RasterizeCudaContext()

    def _tensors(self) -> Dict[str, Tensor]:
        return {
            "V": self.V,
            "F": self.F,
            "uv_co": self.uv_co,
            "texture": self.texture._tensor,
            "opengl_conversion": self.opengl_conversion,
        }

    def _apply_tensors(self, tensor_dict: Dict[str, Tensor]):
        self.V = tensor_dict["V"]
        self.F = tensor_dict["F"]
        self.uv_co = tensor_dict["uv_co"]
        self.texture._tensor = tensor_dict["texture"]
        self.opengl_conversion = tensor_dict["opengl_conversion"]

    def to(self, *args, **kwargs) -> TexturedMesh:
        super().to(*args, **kwargs)
        device = self.device
        if device.type == "cuda":
            self.ctx = dr.RasterizeCudaContext()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "V": self.V,
            "F": self.F,
            "uv_co": self.uv_co,
            "texture": self.texture,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TexturedMesh:
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

    def camera_to_nvdiffrast(self, camera):
        device = self.device
        dtype = camera.w2c.dtype

        # Rotate the camera's orbital coordinate system:
        #
        # camera's default position:
        #     (0, 0, -r)
        #
        # becomes:
        #     (0, -r, 0)
        #
        # This is a +90° rotation around world X.
        world_adjust = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
            device=device,
        )

        # Your camera convention:
        #   +X right
        #   +Y down
        #   +Z forward
        #
        # OpenGL:
        #   +X right
        #   +Y up
        #   -Z forward
        gl_conversion = torch.diag(
            torch.tensor(
                [1.0, -1.0, -1.0, 1.0],
                dtype=dtype,
                device=device,
            )
        )

        fx = camera.Fx
        fy = camera.Fx

        projection = torch.zeros(
            (4, 4),
            dtype=dtype,
            device=device,
        )

        projection[0, 0] = 2 * fx / camera.W
        projection[1, 1] = -2 * fy / camera.H

        zn = camera.Zn
        zf = camera.Zf

        projection[2, 2] = -(zf + zn) / (zf - zn)
        projection[2, 3] = -2 * zf * zn / (zf - zn)
        projection[3, 2] = -1

        return projection @ gl_conversion @ camera.w2c @ world_adjust

    @property
    def VH(self) -> Float[Tensor, "V 4"]:
        return torch.cat(
            [self.V, torch.ones(self.V.shape[0], 1, device=self.V.device)], dim=1
        )

    def rasterize(self, camera: Camera) -> Float[Tensor, "B 4 H W"]:
        # breakpoint()
        pos_clip = self.VH @ self.camera_to_nvdiffrast(camera).T
        # breakpoint()
        rast, _ = dr.rasterize(self.ctx, pos_clip[None], self.F, [camera.H, camera.W])
        uv_img, _ = dr.interpolate(self.uv_co[None], rast, self.F)
        rgb = dr.texture(
            self.texture._tensor[:, :3].permute(0, 2, 3, 1).contiguous(),
            uv_img,
            filter_mode="linear",
            boundary_mode="wrap",
        )
        rgba = torch.cat(
            [rgb, torch.ones((*rgb.shape[:3], 1), device=rgb.device)], dim=-1
        )
        rgba = torch.where(rast[..., 3:4] > 0, rgba, torch.zeros_like(rgba))
        return rgba.permute(0, 3, 1, 2)

    @classmethod
    def from_mesh_data(
        cls,
        V: Float[Tensor, "V"],
        F: Float[Tensor, "F"],
        uv_co: Float[Tensor, "V 2"],
        texture: Splimage,
        unit_box: bool = True,
        **kwargs,
    ) -> TexturedMesh:
        if unit_box:
            V_min, V_max = V.min(dim=0).values, V.max(dim=0).values
            V_extent = V_max - V_min
            V_center = (V_max + V_min) / 2
            V = (V - V_center) / V_extent.max()
        V = torch.stack([V[:, 1], V[:, 0], -V[:, 2]], dim=1)
        return TexturedMesh(V, F, uv_co, texture)
