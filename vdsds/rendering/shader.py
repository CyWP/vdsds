from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .context import RenderContext
from .utils import attr_map


class Shader(ABC):
    @abstractmethod
    def apply(self, ctx: RenderContext) -> dict[str, Float[Tensor, "B H W C"]]:
        pass

    def __call__(self, ctx: RenderContext) -> dict[str, Float[Tensor, "B H W C"]]:
        return self.apply(ctx)


class MeshAttrMap(Shader):
    def __init__(self, attr: str, out_key: str | None = None):
        self.attr = attr
        self.out_key = attr if out_key is None else out_key

    def apply(self, ctx: RenderContext) -> dict[str, Float[Tensor, "B H W C"]]:
        return {self.out_key: attr_map(ctx, getattr(ctx.mesh, self.attr))}


class Albedo(MeshAttrMap):
    def __init__(self):
        super().__init__("texture", "render")


class Normal(MeshAttrMap):
    def __init__(self):
        super().__init__("vertex_normals_normalized", "normal")


class BranchShader(Shader, ABC):
    def __init__(
        self,
        apply_to: set[str] | None = None,
    ):
        self.apply_to = apply_to

    @abstractmethod
    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        pass

    def __call__(self, ctx: RenderContext) -> dict[str, Float[Tensor, "B H W C"]]:
        out = {}
        at = self.apply_to
        for k, v in ctx.out.items():
            if at is None or k in at:
                out[k] = self.apply(ctx, v)
            else:
                out[k] = v
        return out


class Clamp(BranchShader):
    def __init__(
        self,
        min: float = 0.0,
        max: float = 1.0,
        apply_to: set[str] | None = None,
    ):
        self.apply_to = apply_to
        self.min = min
        self.max = max

    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        return img.clamp(min=self.min, max=self.max)


class Alpha(BranchShader):
    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        B, H, W, C = img.shape
        if C != 3:
            raise ValueError("Can only add alpha to image with 3 channels.")
        imga = torch.cat([img, torch.ones_like(img[..., 2:])], dim=-1)
        imga = torch.where(ctx.rast[..., 3:4] > 0, imga, torch.zeros_like(imga))
        return imga


class BackgroundColor(BranchShader):
    def __init__(
        self,
        color: Float[Tensor, 3],
        apply_to: set[str] | None = None,
    ):
        super().__init__(apply_to=apply_to)
        self.color = color

    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        B, H, W, C = img.shape
        if C != 4:
            raise ValueError("Can only add background to image with 4 channels.")
        alpha = img[..., 3:]
        rgb = img[..., :3]
        return alpha * rgb + (1 - alpha) * self.color[None, None, None].expand(
            rgb.shape
        )


class LambdaShader(BranchShader):
    def __init__(
        self,
        func: Callable[
            [RenderContext, Float[Tensor["B H W C"]], Float[Tensor, "B H W C"]]
        ],
        apply_to: set[str] | None = None,
    ):
        super().__init__(apply_to=apply_to)
        self.func = func

    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        return self.func(ctx, img)


class LambertShader(BranchShader):
    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        mesh = ctx.mesh
        normal = ctx.out.get("normal", attr_map(ctx, mesh.vertex_normals_normalized))
        pos = attr_map(ctx, mesh.V)
        light = ctx.light
        ray = light.origin[None, None, None] - pos
        return img * light.strength * (ray * normal).sum(dim=-1, keepdim=True)


class HalfLambertShader(BranchShader):
    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        mesh = ctx.mesh
        normal = ctx.out.get("normal", attr_map(ctx, mesh.vertex_normals_normalized))
        pos = attr_map(ctx, mesh.V)
        light = ctx.light
        ray = light.origin[None, None, None] - pos
        return (
            img * light.strength * ((ray * normal).sum(dim=-1, keepdim=True) / 2 + 0.5)
        )


class SoftLambertShader(BranchShader):
    def __init__(self, apply_to: set[str] | None = None, beta: int = 6.0):
        super().__init__(apply_to=apply_to)
        self.beta = beta

    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        mesh = ctx.mesh
        normal = ctx.out.get("normal", attr_map(ctx, mesh.vertex_normals_normalized))
        pos = attr_map(ctx, mesh.V)
        light = ctx.light
        ray = light.origin[None, None, None] - pos
        return (
            img
            * light.strength
            * F.softplus((ray * normal).sum(dim=-1, keepdim=True), beta=self.beta)
        )


class Antialias(BranchShader):
    def apply(
        self, ctx: RenderContext, img: Float[Tensor, "B H W C"]
    ) -> Float[Tensor, "B H W C"]:
        return dr.antialias(img, ctx.rast, ctx.pos_clip, ctx.mesh.F)
