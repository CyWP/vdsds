import torch


from ..utils.harmonics import SphericalHarmonic
from ..utils.camera import Camera
from .base import Deformation
from .mesh import Mesh
from .splat_mesh import SplatMesh


class SplatMeshDeformation(Deformation):
    def __init__(self, model: SplatMesh, degree: int = 2):
        super().__init__(model)
        self.V_deform = SphericalHarmonic(
            degree=degree, num_dims=3, batch_size=model.mesh.num_V
        )
        self.colors_deform = SphericalHarmonic(
            degree=degree, num_dims=3, batch_size=len(SplatMesh)
        )
        self.register_buffer("degree", degree)

    def deformed(self, camera: Camera) -> SplatMesh:
        camera_loc = camera.location.unsqueeze(0)
        m = self.model
        delta = m.centroid - camera_loc
        V_disp = self.V_deform.from_cartesian(delta)
        colors_disp = self.colors_deform.from_cartesian(delta)
        return SplatMesh(
            mesh=Mesh(m.mesh.V + V_disp),
            faces=m.faces,
            barys=m.barys,
            disps=m.disps,
            scales=m.scales,
            rots=m.rots,
            colors=(m.colors + colors_disp).clamp(0, 1),
            sh_degree=m.sh_degree,
        )
