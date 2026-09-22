from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..representations.mesh import Mesh
from ..utils.camera import Camera
from ..utils.poisson_system import PoissonSystem
from .base import Deformation


class FullJacobianDeformation(Deformation):
    def __init__(
        self,
        model: Mesh,
        **kwargs,
    ):
        super().__init__(model)
        device = model.V.device
        dtype = model.V.dtype
        self.poisson = PoissonSystem.from_mesh(model.V, model.F)
        self.J_deform = nn.Parameter(
            torch.eye(3, device=device, dtype=dtype)[None]
            .repeat(model.num_F, 1, 1)
            .contiguous()
            * 0.0
        )
        self._cached = False

    def to(self, device: torch.device | str):
        super().to(device)
        self.poisson = self.poisson.to(device)
        return self

    def _dict_data(self) -> dict[str, Any]:
        return {**super()._dict_data(), "J_deform": self.J_deform}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FullJacobianDeformation:
        model = Mesh.from_dict(data["model"])
        instance = cls(model=model)
        instance.J_deform = nn.Parameter(data["J_deform"])
        return instance

    def cache_solver(self):
        self.J_src = self.poisson.jacobians_from_vertices(self.model.V[None])
        self._cached = True

    def jacobians_3d(
        self, delta: Float[Tensor,] | None = None
    ) -> Float[Tensor, "B F 3 3"]:
        return (
            self.J_deform
            + torch.eye(3, device=self.J_deform.device, dtype=self.J_deform.dtype)[None]
        )[None]

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
        J_transformed = torch.einsum("bfij,bfjk->bfik", self.jacobians_3d(), self.J_src)
        V_new = self.poisson.solve_poisson(J_transformed)[0]
        return Mesh(V=V_new, F=self.model.F)
