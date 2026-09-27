from jaxtyping import Float
from torch import Tensor

from .rendering.camera import Camera


class Rasterizable:
    def rasterize(self, camera: Camera) -> Float[Tensor, "B 4 H W"]:
        raise NotImplementedError()
