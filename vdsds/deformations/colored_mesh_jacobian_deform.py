from __future__ import annotations

import torch
from torch import Tensor

from ..representations.vertextured_mesh import VerTexturedMesh
from ..utils.camera import Camera
from ..utils.harmonics import SphericalHarmonic
from ..utils.poisson_system import poisson_system_matrices_from_mesh
from .base import Deformation


class ColoredMeshJacobianDeformation(Deformation):
    def __init__(self, model: VerTexturedMesh, degree: int = 1, start_degree: int = 1):
        super().__init__(model)
        self.register_buffer("degree", torch.tensor(degree))
        self.psm = poisson_system_matrices_from_mesh(
            V=model.V.cpu().numpy(), F=model.F.cpu().numpy()
        )
        self.solver = self.psm.create_poisson_solver()
        self.J_deform = SphericalHarmonic(
            degree, 9, model.num_F, start_degree=start_degree
        )
        self.color_deform = SphericalHarmonic(
            degree, 3, model.num_V, start_degree=start_degree
        )
        self._cached = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.solver = self.solver.to(*args, **kwargs)
        return self

    def cache_solver(self):
        self.J_src = self.solver.jacobians_from_vertices(self.model.V[None])
        self._cached = True

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Tensor]
    ) -> ColoredMeshJacobianDeformation:
        model_keys = {}
        v_deform_keys = {}
        colors_deform_keys = {}
        direct_keys = {}

        for key, value in state_dict.items():
            if key.startswith("model."):
                model_keys[key[6:]] = value
            elif key.startswith("V_deform."):
                v_deform_keys[key[9:]] = value
            else:
                direct_keys[key] = value

        model = VerTexturedMesh.from_state_dict(model_keys)
        degree = int(direct_keys.get("degree", 2))

        instance = cls(model=model, degree=degree)
        instance.V_deform = SphericalHarmonic.from_state_dict(v_deform_keys)
        instance.colors_deform = SphericalHarmonic.from_state_dict(colors_deform_keys)
        return instance

    def deformed(self, camera: Camera) -> VerTexturedMesh:
        if not self._cached:
            self.cache_solver()
        camera_loc = camera.location.unsqueeze(0)
        m = self.model
        delta = m.centroid - camera_loc
        J_disp = (
            self.J_deform.from_cartesian(delta).reshape(-1, 3, 3)
            + torch.eye(3, device=self.device)[None]
        )
        J_transformed = torch.einsum("bfij,bfjk->bfik", J_disp[None], self.J_src)
        V_new = self.solver.solve_poisson(J_transformed)[0]
        color_disp = self.color_deform.from_cartesian(delta)
        return VerTexturedMesh(V=V_new, F=m.F, color=m.color + color_disp)
