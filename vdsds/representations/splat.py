from __future__ import annotations
import torch

from typing import Any, Dict, Sequence
from torch import Tensor
from jaxtyping import Float

from gsplat import rasterization

from ..utils.camera import Camera

from .base import Model


class Splat(Model):
    def __init__(
        self,
        means: Float[Tensor, "N 3"],
        quats: Float[Tensor, "N 4"],
        scales: Float[Tensor, "N 3"],
        opacities: Float[Tensor, "N 1"],
        colors: Float[Tensor, "N 1 C"],
        sh_degree: int,
    ):
        super().__init__()
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.colors = colors
        self.sh_degree = sh_degree
        self.alpha_cutoff = 0.0

    def _tensors(self) -> Dict[str, Tensor]:
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "colors": self.colors,
        }

    def _apply_tensors(self, tensor_dict: Dict[str, Tensor]):
        self.means = tensor_dict["means"]
        self.quats = tensor_dict["quats"]
        self.scales = tensor_dict["scales"]
        self.opacities = tensor_dict["opacities"]
        self.colors = tensor_dict["colors"]

    def copy(self) -> Splat:
        return Splat(
            means=self.means.clone(),
            quats=self.quats.clone(),
            scales=self.scales.clone(),
            opacities=self.opacities.clone(),
            colors=self.colors.clone(),
            sh_degree=self.sh_degree,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "colors": self.colors,
            "sh_degree": self.sh_degree,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Splat:
        return cls(
            means=data["means"],
            quats=data["quats"],
            scales=data["scales"],
            opacities=data["opacities"],
            colors=data["colors"],
            sh_degree=int(data["sh_degree"]),
        )

    @staticmethod
    def combine(splats: Sequence[Splat]) -> Splat:
        if len(splats) == 0:
            raise ValueError("Cannot pass empty iterable of splats.")
        if len(splats) == 1:
            return splats[0]
        new_sh_degree = splats[0].sh_degree
        if not all(splat.sh_degree == new_sh_degree for splat in splats):
            raise ValueError(
                "Cannot combine splats with different spherical harmonic degrees."
            )
        new_means = torch.cat([splat.means for splat in splats], dim=0)
        new_Q = torch.cat([splat.quats for splat in splats], dim=0)
        new_scales = torch.cat([splat.scales for splat in splats], dim=0)
        new_opacities = torch.cat([splat.opacities for splat in splats], dim=0)
        new_colors = torch.cat([splat.colors for splat in splats], dim=0)
        return Splat(
            new_means,
            new_Q,
            new_scales,
            new_opacities,
            new_colors,
            new_sh_degree,
        )

    def __len__(self) -> int:
        return self.means.shape[0]

    def centroid(self) -> Float[Tensor, "3"]:
        return torch.mean(self.means, dim=0)

    def rasterize(self, camera: Camera) -> Float[Tensor, "1 4 H W"]:
        color, alpha, _ = rasterization(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.colors,
            camera.w2c.unsqueeze(0).contiguous(),
            camera.K.unsqueeze(0).contiguous(),
            camera.W,
            camera.H,
            render_mode="RGB",
            sh_degree=self.sh_degree,
            backgrounds=None,
        )
        return torch.cat([color, alpha], dim=-1).permute(0, 3, 1, 2)
