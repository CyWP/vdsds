from __future__ import annotations

from typing import ClassVar

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float
from torch import Tensor

from ..representations.mesh import Mesh
from .camera import Camera
from .context import RenderContext
from .conventions import NVDIFFRAST_CONVERSION_MTX, OPENGL_CONVERSION_MTX, UP
from .light import LightSource
from .shader import (
    Albedo,
    Alpha,
    Antialias,
    BackgroundColor,
    Clamp,
    Normal,
    Shader,
    SoftLambertShader,
)


class Renderer:
    _default_device: ClassVar[str] = "cuda:0"
    _default_dtype = torch.float32

    def __init__(
        self,
        shaders: list[Shader],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is None:
            device = torch.device(self._default_device)
        if device.type != "cuda":
            raise ValueError("Shaders only support cuda GPUs.")
        if dtype is None:
            dtype = self._default_dtype
        if len(shaders) == 0:
            raise ValueError("Renderer needs at least one shader to output image.")
        self.nvdr_ctx = dr.RasterizeCudaContext()
        self.opengl_conversion = torch.tensor(
            OPENGL_CONVERSION_MTX,
            dtype=dtype,
            device=device,
        )
        self.nvdiffrast_conversion = torch.tensor(
            NVDIFFRAST_CONVERSION_MTX,
            dtype=dtype,
            device=device,
        )
        self.up = torch.tensor(UP, device=device, dtype=dtype)
        self.shaders = shaders

    @classmethod
    def basic(
        cls,
        bg_color: Float[Tensor, 3],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Renderer:
        shaders = [
            Albedo(),
            Normal(),
            SoftLambertShader(apply_to={"render"}),
            Alpha(apply_to={"render"}),
            BackgroundColor(
                color=bg_color,
                apply_to={"render"},
            ),
            Clamp(apply_to={"render"}),
            Antialias(apply_to={"render"}),
        ]
        return Renderer(shaders, device=device, dtype=dtype)

    def camera_to_nvdiffrast(self, camera):
        return (
            camera.projection
            @ self.opengl_conversion
            @ camera.w2c
            @ self.nvdiffrast_conversion
        )

    def render(
        self, mesh: Mesh, camera: Camera, light: LightSource, **kwargs
    ) -> dict[str, Float[Tensor, "B C H W"]]:
        ctx = RenderContext(mesh=mesh, camera=camera, light=light, **kwargs)
        ctx.pos_clip = (
            torch.cat([ctx.mesh.V, torch.ones_like(ctx.mesh.V[:, 0:1])], dim=-1)
            @ self.camera_to_nvdiffrast(camera).T
        )
        ctx.rast, _ = dr.rasterize(
            self.nvdr_ctx, ctx.pos_clip[None], mesh.F, [camera.H, camera.W]
        )
        ctx.out = {}
        for shader in self.shaders:
            ctx.out.update(**shader(ctx))
        for k, v in ctx.out.items():
            ctx.out[k] = v.permute(0, 3, 1, 2)
        return ctx.out

    def __call__(
        self, mesh: Mesh, camera: Camera, light: LightSource, **kwargs
    ) -> dict[str, Float[Tensor, "B C H W"]]:
        return self.render(mesh, camera, light, **kwargs)
