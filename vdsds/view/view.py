import numpy as np
from typing import Optional, Callable

from ..rasterizable import Rasterizable
from ..utils.timer import TimedJob
from ..utils.camera import Camera
from .obj_view import ObjViewer
from .window import AppView
from .keymap import KeyHandler


class View:
    def __init__(
        self,
        obj: Rasterizable,
        camera: Camera,
        fps: int = 24,
        on_close: Optional[Callable] = None,
    ):
        self.window = AppView()
        self.viewer = ObjViewer(obj)
        self.keys = KeyHandler()
        self.on_close = on_close
        self._connect()
        self.frame_retriever = TimedJob()
        self.frame_retriever.start(self.update, fps)

    def _connect(self):
        w = self.window.window
        w.key_pressed.connect(self.keys.pressed)
        w.key_released.connect(self.keys.released)
        v = w.viewport
        v.drag_direction.connect(self.viewer.mouse_drag)
        v.scrolled.connect(self.viewer.scroll)
        v.left_clicked.connect(
            lambda x, y: self.viewer.left_click(x, y, list(self.keys))
        )
        v.right_clicked.connect(
            lambda x, y: self.viewer.right_click(x, y, list(self.keys))
        )
        if self.on_close is not None:
            w.closed.connect(self.on_close)

    def update(self):
        image = self.viewer.get_render(*self.window.dimensions)
        image = (image.permute(0, 2, 3, 1).contiguous().numpy() * 255).astype(np.uint8)
        self.window.update(image[0])
