from __future__ import annotations

from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from .camera import Camera
from .conventions import UP


def cosine_sample_hemisphere(n: Float[Tensor, 3]) -> Float[Tensor, 3]:
    """Samples a direction from the cosine-weighted hemisphere around n.

    Port of nvdiffmodeling's ``cosine_sample`` (used by MeshUp's camera
    augmentation), torch-only so it respects torch's RNG seeding.

    Args:
        n: Hemisphere axis (any magnitude, normalized internally).

    Returns:
        out: Unit direction, cosine-distributed about n.
    """
    n = n / n.norm().clamp_min(torch.finfo(n.dtype).tiny)
    dx0 = torch.tensor([0.0, n[2], -n[1]], device=n.device)
    dx1 = torch.tensor([-n[2], 0.0, n[0]], device=n.device)
    dx = dx0 if (dx0 * dx0).sum() > (dx1 * dx1).sum() else dx1
    dx = dx / dx.norm()
    dy = torch.linalg.cross(n, dx)
    phi = 2.0 * torch.pi * torch.rand((), device=n.device)
    s = torch.rand((), device=n.device)
    sintheta = (1.0 - s).sqrt()
    return dx * (phi.cos() * sintheta) + dy * (phi.sin() * sintheta) + n * s.sqrt()


class LightSource:
    """
    Just a representation of a point light
    """

    _up: ClassVar = [UP[0], UP[2], UP[1]]

    def __init__(
        self,
        origin: Float[Tensor, 3] | None = None,
        strength: Float[Tensor, ""] | None = None,
    ):
        self.origin = torch.tensor(self._up) if origin is None else origin
        self.strength = torch.tensor(1.5) if strength is None else strength

    @property
    def device(self) -> torch.device:
        return self.origin.device

    @property
    def dtype(self) -> torch.dtype:
        return self.origin.dtype

    def to(self, device: torch.device | str) -> LightSource:
        self.origin = self.origin.to(device)
        self.strength = self.strength.to(device)
        return self

    def requires_grad_(self, mode: bool) -> LightSource:
        self.origin.requires_grad_(mode)
        self.strength.requires_grad_(mode)
        return self

    def copy(self) -> LightSource:
        """
        Return a copy of the object.
        """
        return LightSource(self.origin.clone(), self.strength.clone())

    @classmethod
    def from_camera(cls, camera: Camera, strength: float | None = None) -> LightSource:
        device = camera.device
        o = camera.location[[0, 2, 1]] * torch.tensor([-1.0, -1.0, 1.0], device=device)
        return LightSource(o, strength)
