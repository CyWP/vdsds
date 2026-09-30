from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from ..rendering.camera import Camera, CameraCoordinates
from ..rendering.light import LightSource
from ..rendering.renderer import Renderer
from ..utils.config import Config
from ..utils.img import Splimage
from ..utils.quaternion import Quaternion
from ..utils.video import write_video
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
            "window": {
                "view": False,
                "bg_color": [1.0, 1.0, 1.0],  # Color or None
            },
            "path": {
                "model": None,
                "out_dir": "./recordings",
                "file_name": "orbit",
            },
            "camera": {
                "H": 1024,
                "W": 1024,
                "F": 60,
                "radius": 1.5,  # None: auto from model bounding sphere.
            },
            "lighting": {
                "alignment": "up",  # Options: 'camera', 'up', or [x, y, z].
                "strength": 1.5,
            },
            "rotation": {
                "mode": "both",  # Options: 'horizontal', 'vertical', 'both'.
                "frames": 240,
            },
        }
    )

    def run(self):
        cfg = self.config
        out_dir = Path(cfg.path.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.renderer = self.get_renderer()
        video_path = out_dir / f"{cfg.path.file_name}.mp4"
        write_video(self.orbit_all(), path=video_path)
        print(f"Orbit video saved in {video_path}.")

    def orbit_all(self) -> Iterator[Splimage]:
        device = self.device
        cfg = self.config
        mode = cfg.rotation.mode
        if mode not in ("horizontal", "vertical", "both"):
            raise ValueError(f"Invalid orbit mode: '{mode}'.")
        rot_frames = cfg.rotation.frames
        camera = self.get_camera().to(device)
        frame_add = 0
        axis = CameraCoordinates._up.to(device)
        if mode in ("horizontal", "both"):
            for frame in self.orbit_frames(camera, axis, rot_frames):
                yield frame
            frame_add += rot_frames
        if mode in ("vertical", "both"):
            axis = axis[[2, 0, 1]]
            for frame in self.orbit_frames(camera, axis, rot_frames):
                yield frame
        return

    def orbit_frames(
        self, camera: Camera, axis: Float[Tensor, "3"], frames: int
    ) -> Iterator[Splimage]:
        Q = Quaternion.from_axis_angle(
            axis=axis, angle=torch.tensor(2 * torch.pi / frames, device=axis.device)
        ).to(camera.device)
        for _ in range(frames):
            render = self.render(camera)
            camera.co.Q *= Q
            yield render

    def get_renderer(self) -> Renderer:
        cfg = self.config
        return Renderer.basic(
            bg_color=torch.tensor(cfg.window.bg_color, device=self.device),
            device=self.device,
        )

    def render(self, camera: Camera, bg: Float[Tensor, "3"] | None = None) -> Splimage:
        light = self.get_light(camera)
        rgb = self.renderer.render(self.model.deformed(camera), camera, light)[
            "render"
        ][:, :3]
        return Splimage(rgb)

    def get_camera(self) -> Camera:
        cfg = self.config.camera
        co = CameraCoordinates(
            Q=Quaternion.from_axis_angle(
                CameraCoordinates._up, torch.tensor(torch.pi / 2)
            ),
            radius=cfg.radius,
        )
        return Camera(H=cfg.H, W=cfg.W, F=cfg.F, co=co)

    def get_light(self, camera: Camera) -> LightSource:
        cfg = self.config.lighting
        device = camera.device
        align = cfg.alignment
        strength = torch.tensor(cfg.strength)
        if align == "camera":
            return LightSource.from_camera(camera, strength).to(device)
        elif align == "up":
            return LightSource(strength=strength).to(device)
        elif isinstance(align, (list, tuple)) and len(align) == 3:
            return LightSource(
                torch.tensor(align, device=device, dtype=torch.float32), strength
            ).to(device)
        raise ValueError(
            f"Alignment '{align}' is invalid for lighting alignment config; "
            "expected 'camera', 'up', or a vector of length 3."
        )
