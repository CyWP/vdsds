from __future__ import annotations

import torch

from typing import Dict
from torch import Tensor

from ..utils.harmonics import SphericalHarmonic
from ..utils.camera import Camera
from .base import Deformation
from ..representations.mesh import Mesh
from ..representations.splat_mesh import SplatMesh


class SplatMeshDeformation(Deformation):
    def __init__(self, model: SplatMesh, degree: int = 2):
        super().__init__(model)
        self.V_deform = SphericalHarmonic(
            degree=degree, num_dims=3, batch_size=model.mesh.num_V
        )
        self.colors_deform = SphericalHarmonic(
            degree=degree, num_dims=3, batch_size=len(model)
        )
        self.register_buffer("degree", torch.tensor(degree))

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> SplatMeshDeformation:
        model_keys = {}
        v_deform_keys = {}
        colors_deform_keys = {}
        direct_keys = {}

        for key, value in state_dict.items():
            if key.startswith("model."):
                model_keys[key[6:]] = value
            elif key.startswith("V_deform."):
                v_deform_keys[key[9:]] = value
            elif key.startswith("colors_deform."):
                colors_deform_keys[key[14:]] = value
            else:
                direct_keys[key] = value

        model = SplatMesh.from_state_dict(model_keys)
        degree = int(direct_keys.get("degree", 2))

        instance = cls(model=model, degree=degree)
        instance.V_deform = SphericalHarmonic.from_state_dict(v_deform_keys)
        instance.colors_deform = SphericalHarmonic.from_state_dict(colors_deform_keys)
        return instance

    def deformed(self, camera: Camera) -> SplatMesh:
        camera_loc = camera.location.unsqueeze(0)
        m = self.model
        delta = m.centroid - camera_loc
        V_disp = self.V_deform.from_cartesian(delta)
        colors_disp = self.colors_deform.from_cartesian(delta)
        return SplatMesh(
            mesh=Mesh(m.mesh.V + V_disp, m.mesh.F),
            faces=m.faces,
            barys=m.barys,
            disps=m.disps,
            scales=m.scales,
            rots=m.rots,
            colors=(m.colors + colors_disp),
            sh_degree=m.sh_degree,
        )
