import numpy as np
import traceback
from typing import Optional, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication

from ..rasterizable import Rasterizable
from ..utils.timer import TimedJob
from ..utils.camera import Camera
from .obj_view import ObjViewer
from .window import AppView
from .keymap import KeyHandler


class View(QApplication):
    frame_ready = Signal(np.ndarray)

    def __init__(
        self,
        obj: Rasterizable,
        camera: Optional[Camera] = None,
        fps: int = 24,
        on_close: Optional[Callable] = None,
    ):
        super().__init__()
        self.window = AppView()
        self.viewer = ObjViewer(obj, camera=camera)
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

    def close(self):
        self.frame_retriever.stop()
        self.window.window.close()

    def update(self):
        try:
            image = self.viewer.get_render(*self.window.dimensions)
            image = (image.permute(0, 2, 3, 1).cpu().contiguous().numpy() * 255).astype(
                np.uint8
            )
            self.frame_ready.emit(image[0])
        except Exception:
            traceback.print_exc()
