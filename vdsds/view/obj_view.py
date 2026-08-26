import torch
import math

from typing import Optional, List
from jaxtyping import Float
from torch import Tensor

from ..rasterizable import Rasterizable
from ..utils.camera import Camera
from .keymap import K_SHIFT, K_CTRL


class ObjViewer:
    def __init__(
        self,
        obj: Rasterizable,
        camera: Optional[Camera] = None,
        sensitivity: float = 60.0,
    ):
        self.obj = obj
        self.camera = (
            Camera().to(obj.device) if camera is None else camera.to(obj.device)
        )
        self.sensitivity = sensitivity
        self.rot_x: int = 0
        self.rot_y: int = 0
        self.tran_x: int = 0
        self.tran_y: int = 0
        self.roll_x: int = 0
        self.roll_y: int = 0

    @torch.no_grad()
    def get_render(self, H: int, W: int) -> Float[Tensor, "B 4 H W"]:
        if H != self.camera.H or W != self.camera.W:
            self.camera.set_window_size(H, W)
        self.check_rotation()
        self.check_roll()
        self.check_translation()
        render = self.obj.rasterize(self.camera)
        # render = Float[Tensor, "B 4 H W"].fill_background(render, self.bg_color)
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

    def mouse_drag(self, x: int, y: int, keys: List[int]):
        if K_SHIFT in keys:
            self.tran_x += x
            self.tran_y += y
        elif K_CTRL in keys:
            self.roll_x += x
            self.roll_y += y
        else:
            self.rot_x += x
            self.rot_y += y

    def left_click(self, x: int, y: int, keys: List[int]):
        pass

    def right_click(self, x: int, y: int, keys: List[int]):
        pass

    def scroll(self, x: int, keys: List[int] = None):
        self.camera.translate_depth(x / 600)
