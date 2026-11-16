import traceback
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication

from ..deformations.base import Deformation
from ..rendering.camera import Camera
from ..rendering.light import LightSource
from ..representations.base import Model
from ..utils.timer import TimedJob
from .keymap import KeyHandler
from .obj_view import ObjViewer
from .window import AppView


def draw_text_overlay(image: np.ndarray, text: str) -> np.ndarray:
    """Draws a top-left text overlay onto an HxWx4 uint8 frame.

    Black text with a white offset shadow, so it stays readable on any
    background. Returns the modified array (in place).
    """
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for dx, dy, color in ((1, 1, (255, 255, 255)), (0, 0, (0, 0, 0))):
        draw.multiline_text((8 + dx, 8 + dy), text, fill=color)
    return np.asarray(pil)


class View(QApplication):
    frame_ready = Signal(np.ndarray)

    def __init__(
        self,
        obj: Model | Deformation,
        light: LightSource | None = None,
        camera: Camera | None = None,
        fps: int = 24,
        on_close: Callable | None = None,
        overlay: Callable[[Camera], str | None] | None = None,
    ):
        """Live viewer for a model or deformation.

        Args:
            obj: Model or deformation to display.
            light: Optional fixed light source.
            camera: Optional initial camera.
            fps: Frame rate of the render/update timer.
            on_close: Callback invoked when the window is closed.
            overlay: Optional per-frame text hook, called with the
                viewer's current camera; the returned string is drawn on
                the frame (multi-line supported). Returning None or ''
                draws nothing.
        """
        super().__init__()
        self.window = AppView()
        self.viewer = ObjViewer(obj, camera=camera, light=light)
        self.keys = KeyHandler()
        self.on_close = on_close
        self.overlay = overlay
        self._connect()
        self.frame_retriever = TimedJob()
        self.frame_retriever.start(self.update, fps)

    def _connect(self):
        w = self.window.window
        w.key_pressed.connect(self.keys.pressed)
        w.key_released.connect(self.keys.released)
        v = w.viewport
        v.drag_direction.connect(
            lambda x, y: self.viewer.mouse_drag(x, y, list(self.keys))
        )
        v.scrolled.connect(lambda x: self.viewer.scroll(x, list(self.keys)))
        v.left_clicked.connect(
            lambda x, y: self.viewer.left_click(x, y, list(self.keys))
        )
        v.right_clicked.connect(
            lambda x, y: self.viewer.right_click(x, y, list(self.keys))
        )
        if self.on_close is not None:
            w.closed.connect(self.on_close)
        self.frame_ready.connect(self.window.update)
        self.window.window.view_deformation_btn.toggled.connect(
            lambda checked: setattr(self.viewer, "view_deformed", checked)
        )
        self.window.window.record_btn.toggled.connect(self._toggle_recording)

    def _toggle_recording(self, checked: bool):
        if checked:
            self.viewer.start_recording()
        else:
            path = self.viewer.stop_recording()
            if path:
                print(f"Recording saved to {path}")

    def close(self):
        self.frame_retriever.stop()
        self.window.window.close()

    def update(self):
        try:
            image = self.viewer.get_render(*self.window.dimensions)
            image = (image.permute(0, 2, 3, 1).cpu().contiguous().numpy() * 255).astype(
                np.uint8
            )
            if self.overlay is not None:
                text = self.overlay(self.viewer.camera)
                if text:
                    image = draw_text_overlay(image[0], text)
            else:
                image = image[0]
            self.frame_ready.emit(image)
        except Exception:
            traceback.print_exc()
