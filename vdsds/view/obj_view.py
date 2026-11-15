import math
import time
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from ..deformations.base import Deformation
from ..rendering import Camera, LightSource, Renderer
from ..rendering.shader import (
    Albedo,
    Alpha,
    Antialias,
    Clamp,
    HalfLambertShader,
    Normal,
)
from ..rendering.view_state import Overlay
from ..representations.base import Model
from ..utils.img import ImgUtils, Splimage
from .keymap import K_CTRL, K_SHIFT


class ObjViewer:
    def __init__(
        self,
        obj: Model | Deformation,
        camera: Camera | None = None,
        light: LightSource | None = None,
        sensitivity: float = 2.0,
        bg_color: list[float] | None = None,
    ):
        self.obj = obj
        self.camera = (
            Camera().to(obj.device) if camera is None else camera.to(obj.device)
        )
        self.light = (
            LightSource().to(obj.device) if light is None else light.to(obj.device)
        )
        if bg_color is None:
            bg_color = [1.0, 1.0, 1.0]
        self.bg_color = torch.tensor(bg_color, device=self.camera.device)
        # Half-Lambert (cosine remapped to [0, 1]) keeps the live view
        # gentle and editing-friendly; other Renderer.basic callers
        # (orbit, views) keep the default soft shading.
        self.renderer = Renderer.basic(self.bg_color, shading="half")
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

    def _evaluated_model(self) -> Model:
        """The model (deformed per the current view, when applicable)."""
        if isinstance(self.obj, Deformation):
            if self.view_deformed:
                return self.obj.deformed(self.camera)
            return self.obj.model
        return self.obj

    @torch.no_grad()
    def get_render(
        self, H: int, W: int, overlay: Overlay | None = None
    ) -> Float[Tensor, "B 4 H W"]:
        if H != self.camera.H or W != self.camera.W:
            self.camera.set_window_size(H, W)
        self.check_rotation()
        self.check_roll()
        self.check_translation()
        model = self._evaluated_model()
        render = self.renderer(model, self.camera, self.light)["render"]
        if overlay is not None:
            render = overlay.overlay(render)
        self._save_frame(render)
        return render

    @property
    def transparent_renderer(self) -> Renderer:
        """Background-less variant of the viewer's render chain.

        Same half-Lambert chain minus the BackgroundColor step, so the
        rasterizer alpha channel survives (object pixels alpha 1,
        background alpha 0) for transparent exports.
        """
        if not hasattr(self, "_transparent_renderer"):
            shaders = [
                Albedo(),
                Normal(),
                HalfLambertShader(apply_to={"render"}),
                Alpha(apply_to={"render"}),
                Clamp(apply_to={"render"}),
                Antialias(apply_to={"render"}),
            ]
            self._transparent_renderer = Renderer(
                shaders, device=self.camera.device
            )
        return self._transparent_renderer

    @torch.no_grad()
    def get_transparent_render(self, H: int, W: int) -> Float[Tensor, "B 4 H W"]:
        """RGBA render without background compositing.

        Same camera, light and shading as :meth:`get_render`, but the
        BackgroundColor step is skipped: background pixels carry alpha 0
        and the returned image is 4-channel. The overlay is not
        composited (it is a viewport aid, not part of the model). Camera
        bookkeeping (window size, rotation/roll/translation deltas) is
        idempotent here, so calling this after a regular
        :meth:`get_render` in the same frame tick reproduces that
        frame's camera state exactly.
        """
        if H != self.camera.H or W != self.camera.W:
            self.camera.set_window_size(H, W)
        self.check_rotation()
        self.check_roll()
        self.check_translation()
        model = self._evaluated_model()
        return self.transparent_renderer(model, self.camera, self.light)["render"]

    def check_rotation(self):
        # Rotate if needed
        if self.rot_x == 0 and self.rot_y == 0:
            return
        x, y = self.rot_x, self.rot_y
        self.rot_x = self.rot_y = 0
        angle = (
            math.sqrt((x / self.camera.W) ** 2 + (y / self.camera.H) ** 2)
            * self.sensitivity
        )
        self.camera.rotate_from_image_space(x, y, angle)

    def check_roll(self):
        # Roll if needed
        if self.roll_x == 0:
            return
        x, y = self.roll_x, self.roll_y
        self.roll_x = self.roll_y = 0
        angle = x / self.camera.W * self.sensitivity
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
