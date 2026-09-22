from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ..representations.mesh import Mesh
from ..utils.camera import Camera
from ..utils.poisson_system import PoissonSystem
from ..utils.spherical_basis import SphericalGaussianBasis
from .base import Deformation


class VDFullJacobianDeformation(Deformation):
    def __init__(
        self,
        model: Mesh,
        num_funcs: int = 8,
        centroid_init: str = "fibonacci",
        overlap: float = 2.0,
        **kwargs,
    ):
        super().__init__(model)
        self.poisson = PoissonSystem.from_mesh(model.V, model.F)
        self.J_deform = SphericalGaussianBasis(
            num_funcs, 9, model.num_F, init=centroid_init, sigma_overlap=overlap
        )
        self._cached = False

    def to(self, device: torch.device | str):
        super().to(device)
        self.poisson = self.poisson.to(device)
        return self

    def _dict_data(self) -> dict[str, Any]:
        return {
            **super()._dict_data(),
            "num_funcs": self.J_deform.num_funcs,
            "J_deform": self.J_deform.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> VDFullJacobianDeformation:
        model = Mesh.from_dict(data["model"])
        instance = cls(model=model, num_funcs=data["num_funcs"])
        instance.J_deform = SphericalGaussianBasis.from_dict(data["J_deform"])
        return instance

    def cache_solver(self):
        self.J_src = self.poisson.jacobians_from_vertices(self.model.V[None])
        self._cached = True

    def jacobians_3d(
        self, delta: Float[Tensor,] | None = None
    ) -> Float[Tensor, "B F 3 3"]:
        if delta is None:
            B, N, C = self.J_deform.weights.shape
            j3d = self.J_deform.weights.permute(1, 0, 2).reshape(N, -1, 3, 3)
        else:
            j3d = self.J_deform.from_cartesian(delta).reshape(1, -1, 3, 3)
        return j3d + torch.eye(3, device=self.device, dtype=j3d.dtype)

    def deformed(self, camera: Camera) -> Mesh:
        """Computes the view dependent deformed mesh.

        Args:
            camera: camera used to evaluate the view dependent deformation.

        Returns:
            The deformed mesh.

        Raises:
            FloatingPointError: if the network predicts NaNs/Infs or the
                poisson solve fails (see the poisson system log).
        """
        if not self._cached:
            self.cache_solver()
        camera_loc = camera.location.unsqueeze(0)
        m = self.model
        delta = m.centroid - camera_loc
        J_transformed = torch.einsum(
            "bfij,bfjk->bfik", self.jacobians_3d(delta), self.J_src
        )
        V_new = self.poisson.solve_poisson(J_transformed)[0]
        return Mesh(V=V_new, F=m.F)
