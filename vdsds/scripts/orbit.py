import math
import time
from pathlib import Path
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.camera import Camera, CameraCoordinates
from ..utils.config import Config
from ..utils.img import Splimage
from ..utils.light import LightSource
from ..utils.quaternion import Quaternion
from .base import ViewableScript


class OrbitFrames(ViewableScript):
    """
    Loads a model (or deformation) without opening the viewer, and renders
    frames of a camera orbiting the model. The camera rotates using a
    :class:`Quaternion` at every frame, and each frame is written as an image
    in the output directory. The orbit can be horizontal, vertical, or both.
    """

    _default_config_overrides: ClassVar[Config] = Config(
        {
            "window": {"view": False},
            "path": {
                "model": None,
                "out_dir": "./recordings",
                "prefix": "frame",
            },
            "camera": {
                "H": 512,
                "W": 512,
                "F": 60,
                "radius": None,  # None: auto from model bounding sphere.
                "mode": "both",  # Options: 'horizontal', 'vertical', 'both'.
                "frames": 60,
                "horizontal_deg": 360.0,
                "vertical_deg": 90.0,
                "pitch_level": 0.0,  # Start elevation for 'horizontal' mode.
                "bg_color": [0.2, 0.2, 0.2],
            },
            "lighting": {
                "alignment": "up",  # Options: 'camera', 'up', or [x, y, z].
                "strength": 1.5,
            },
        }
    )

    def run(self):
        cfg = self.config
        out_dir = Path(cfg.path.out_dir) / time.strftime(
            "%Y%m%d_%H%M%S", time.localtime()
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        frames = cfg.camera.frames
        mode = cfg.camera.mode
        if mode not in ("horizontal", "vertical", "both"):
            raise ValueError(f"Invalid orbit mode: '{mode}'.")
        radius = cfg.camera.radius
        if radius is None:
            radius = 2.0 * self.model_radius()

        bg = torch.tensor(cfg.camera.bg_color, device=self.device, dtype=self.dtype)

        n = max(frames - 1, 1)
        if mode in ("horizontal", "both"):
            d_yaw = cfg.camera.horizontal_deg / n
            pitch0 = cfg.camera.pitch_level
        else:
            pitch0 = -cfg.camera.vertical_deg / 2

        d_pitch = 0.0
        if mode in ("vertical", "both"):
            d_pitch = cfg.camera.vertical_deg / n
        if mode == "horizontal":
            d_pitch = 0.0

        axes = torch.stack(
            [
                torch.tensor([0.0, 1.0, 0.0], device=self.device),
                torch.tensor([0.0, 0.0, 1.0], device=self.device),
                torch.tensor([1.0, 0.0, 0.0], device=self.device),
            ]
        )

        Q0 = self.q_step(axes[2], pitch0)
        camera = Camera(
            H=cfg.camera.H,
            W=cfg.camera.W,
            F=cfg.camera.F,
            co=CameraCoordinates(origin=torch.zeros(3), Q=Q0, radius=radius),
        ).to(self.device)
        self._place(camera)
        self.render(camera, bg).save(out_dir / f"{cfg.path.prefix}_0000.png")

        Q_h = self.q_step(axes[0], d_yaw)
        for i in range(1, frames):
            camera.co.Q = Q_h * camera.co.Q
            if d_pitch != 0.0:
                Q_p = self.q_step(axes[2], d_pitch)
                camera.co.Q = Q_p * camera.co.Q
            self._place(camera)
            path = out_dir / f"{cfg.path.prefix}_{i:04d}.png"
            self.render(camera, bg).save(path)

    def q_step(self, axis: Float[Tensor, "3"], deg: float) -> Quaternion:
        """Quaternion rotating `deg` degrees about `axis` (in degrees)."""
        return Quaternion.from_axis_angle(
            axis, torch.tensor(math.radians(deg), device=axis.device)
        )

    def _place(self, camera: Camera):
        """
        Position the camera so its gaze passes through the model center while
        staying at `radius` distance.
        """
        camera.co.origin = (camera.co.radius * camera.R[:, 2].clone()).detach()

    def get_light(self, camera: Camera) -> LightSource:
        cfg = self.config.lighting
        device = camera.device
        align = cfg.alignment
        if align == "camera":
            origin = camera.location.clone()
        elif align == "up":
            origin = torch.tensor(LightSource._up, device=device, dtype=torch.float32)
        elif isinstance(align, (list, tuple)) and len(align) == 3:
            origin = torch.tensor(align, device=device, dtype=torch.float32)
        else:
            raise ValueError(
                f"Alignment '{align}' is invalid for lighting alignment config; "
                "expected 'camera', 'up', or a vector of length 3."
            )
        return LightSource(origin=origin, strength=torch.tensor(cfg.strength))

    def render(self, camera: Camera, bg: Float[Tensor, "3"]) -> Splimage:
        light = self.get_light(camera)
        rgba = self.model.rasterize(camera, light.to(self.device))
        B, C, H, W = rgba.shape
        alpha = rgba[:, 3].unsqueeze(1)
        rgb = alpha * rgba[:, :3] + (1 - alpha) * bg[None, :, None, None].expand(
            B, -1, H, W
        )
        return Splimage(rgb)

    def model_radius(self) -> float:
        model = self.model
        V = getattr(model, "V", None)
        if V is None:
            V = getattr(getattr(model, "model", None), "V", None)
        if V is not None:
            return float(V.norm(dim=-1).max())
        return 1.0
