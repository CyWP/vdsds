from torch import Tensor, nn
from jaxtyping import Float

from ..utils.camera import Camera
from ..rasterizable import Rasterizable
from ..representations.base import Model


class Deformation(nn.Module, Rasterizable):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def __len__(self) -> int:
        return len(self.model)

    def deformed(self, camera: Camera) -> Model:
        raise NotImplementedError()

    def rasterize(self, camera: Camera) -> Float[Tensor, "B 4 H W"]:
        return self.deformed.rasterize(camera)
