import math
import time
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from ..deformations.base import Deformation
from ..representations.base import Model
from ..utils.camera import Camera
from ..utils.img import ImgUtils, Splimage
from ..utils.light import LightSource
from .keymap import K_CTRL, K_SHIFT


class ObjViewer:
    def __init__(
        self,
        obj: Model | Deformation,
        camera: Camera | None = None,
        light: LightSource | None = None,
        sensitivity: float = 60.0,
    ):
        self.obj = obj
        self.camera = (
            Camera().to(obj.device) if camera is None else camera.to(obj.device)
        )
        self.light = (
            LightSource().to(obj.device) if light is None else light.to(obj.device)
        )
        self.sensitivity = sensitivity
        self.rot_x: int = 0
        self.rot_y: int = 0
        self.tran_x: int = 0
        self.tran_y: int = 0
        self.roll_x: int = 0
        self.roll_y: int = 0
        self.view_deformed = True
        self._bg_color: Tensor | None = None
        self._bg_image: Splimage | None = None
        self._recording = False
        self._record_dir: Path | None = None
        self._frame_idx: int = 0

    def start_recording(self, output_dir: str = ".") -> None:
        self._record_dir = Path(output_dir) / f"recording_{time.time():.0f}"
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        self._recording = True

    def stop_recording(self) -> Path | None:
        self._recording = False
        path = self._record_dir
        self._record_dir = None
        self._frame_idx = 0
        return path

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _save_frame(self, render: Float[Tensor, "B 4 H W"]) -> None:
        if not self._recording or self._record_dir is None:
            return
        img = ImgUtils.tensor2pil(render[:, :3].clamp(0, 1))
        path = self._record_dir / f"{self._frame_idx:06d}.png"
        img.save(str(path))
        self._frame_idx += 1

    def set_background(self, bg: Tensor | Splimage | None) -> None:
        if bg is None:
            self._bg_color = None
            self._bg_image = None
        elif isinstance(bg, Splimage):
            self._bg_color = None
            self._bg_image = bg.to(self.obj.device)
        elif isinstance(bg, Tensor):
            self._bg_image = None
            self._bg_color = bg.flatten().to(self.obj.device)
        else:
            raise TypeError(
                f"Expected Tensor, Splimage, or None, got {type(bg).__name__}"
            )

    def _apply_background(
        self, render: Float[Tensor, "B 4 H W"]
    ) -> Float[Tensor, "B 4 H W"]:
        B, C, H, W = render.shape
        rgb = render[:, :3]
        alpha = render[:, 3:4]

        if self._bg_image is not None:
            bg = self._bg_image.image().to(render.device)
            if bg.shape[2] != H or bg.shape[3] != W:
                bg = ImgUtils.resize(bg, H, W)
            if bg.shape[0] < B:
                bg = bg.expand(B, -1, -1, -1)
            bg = bg[:, :3]
        elif self._bg_color is not None:
            bg = self._bg_color.view(1, 3, 1, 1).expand(B, -1, H, W)
        else:
            return render

        composited = (alpha * rgb + (1 - alpha) * bg).clamp(0, 1)
        return torch.cat([composited, torch.ones_like(alpha)], dim=1)

    @torch.no_grad()
    def get_render(self, H: int, W: int) -> Float[Tensor, "B 4 H W"]:
        if H != self.camera.H or W != self.camera.W:
            self.camera.set_window_size(H, W)
        self.check_rotation()
        self.check_roll()
        self.check_translation()
        if isinstance(self.obj, Deformation) and not self.view_deformed:
            render = self.obj.model.rasterize(self.camera, self.light)
        else:
            render = self.obj.rasterize(self.camera, self.light)
        render = self._apply_background(render)
        self._save_frame(render)
        return render

    def check_rotation(self):
        # Rotate if needed
        if self.rot_x == 0 and self.rot_y == 0:
            return
        x, y = self.rot_x, self.rot_y
        self.rot_x = self.rot_y = 0
        angle = (
            math.sqrt((x / self.camera.W) ** 2 + (y / self.camera.H) ** 2)
            * self.sensitivity
            * 0.001
        )
        self.camera.rotate_from_image_space(x, y, angle)

    def check_roll(self):
        # Roll if needed
        if self.roll_x == 0:
            return
        x, y = self.roll_x, self.roll_y
        self.roll_x = self.roll_y = 0
        angle = x / self.camera.W * self.sensitivity * 0.001
        self.camera.roll_from_image_space(angle)

    def check_translation(self):
        # Translate if needed
        if self.tran_x == 0 and self.tran_y == 0:
            return
        x, y = self.tran_x, self.tran_y
        self.tran_x = self.tran_y = 0
        dist = (
            math.sqrt((x / self.camera.W) ** 2 + (y / self.camera.H) ** 2)
            * self.sensitivity
            * 0.02
        )
        self.camera.translate_image_space(x, y, dist)

    def mouse_drag(self, x: int, y: int, keys: list[int]):
        if K_SHIFT in keys:
            self.tran_x += x
            self.tran_y += y
        elif K_CTRL in keys:
            self.roll_x += x
            self.roll_y += y
        else:
            self.rot_x += x
            self.rot_y += y

    def left_click(self, x: int, y: int, keys: list[int]):
        pass

    def right_click(self, x: int, y: int, keys: list[int]):
        pass

    def scroll(self, x: int, keys: list[int] = None):
        self.camera.translate_depth(x / 600)
