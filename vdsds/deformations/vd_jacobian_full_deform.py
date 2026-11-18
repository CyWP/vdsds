from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ..rendering.camera import Camera
from ..representations import get_model
from ..representations.mesh import Mesh
from ..utils.poisson_system import PoissonSystem
from ..utils.spherical_basis import SphericalGaussianBasis
from .base import Deformation


class VDFullJacobianDeformation(Deformation):
    def __init__(
        self,
        model: Mesh,
        num_funcs: int = 8,
        init: str = "fibonacci",
        overlap: float = 2.0,
        normalize: bool = True,
        equi_optim_steps: int | None = None,
        deform_range: float | tuple[float, float] | None = None,
        deform_range_sigma: float = torch.pi / 8,
        **kwargs,
    ):
        super().__init__(model)
        self.poisson = PoissonSystem.from_mesh(model.V, model.F)
        self.J_deform = SphericalGaussianBasis(
            num_funcs,
            9,
            model.num_F,
            init=init,
            sigma_overlap=overlap,
            normalize=normalize,
            equi_optim_steps=equi_optim_steps,
        )
        # Orbit-azimuth gate on the deformation: None deforms the full
        # orbit; a single azimuth deforms only a Gaussian window around
        # that view; a (start, end) pair deforms a contiguous range with
        # one-sided Gaussian tapers (no wrap: start <= end within
        # [-pi, pi]). Azimuth 0 is the front camera (sitting at -Z), +pi/2
        # the +X camera, measured on the camera-position direction
        # (location - centroid). Outside the gate the deformation
        # evaluates to the identity, i.e. the untouched source mesh.
        self.deform_range = deform_range
        self.deform_range_sigma = deform_range_sigma
        self._cached = False

    def view_mask(self, deltas: Float[Tensor, "M 3"]) -> Float[Tensor, "M 1"]:
        """Deform-range gate for camera directions.

        Args:
            deltas: Centroid-minus-camera directions (M, 3); the gate is
            expressed on the opposite (camera-position) direction so that
            azimuth 0 is the front camera.

        Returns:
            out: Per-direction gate values in [0, 1]; 0 fully disables the
            deformation for that view.
        """
        if self.deform_range is None:
            return torch.ones(
                (deltas.shape[0], 1), device=deltas.device, dtype=deltas.dtype
            )
        u = -deltas  # camera-position directions
        theta = torch.atan2(u[:, 0], -u[:, 2])  # (-pi, pi], front = 0

        def wrap(d: Float[Tensor, "M"]) -> Float[Tensor, "M"]:
            return (d + torch.pi) % (2 * torch.pi) - torch.pi

        sig = self.deform_range_sigma
        if torch.is_tensor(self.deform_range) or (
            isinstance(self.deform_range, (list, tuple))
        ):
            a, b = self.deform_range  # type: ignore[misc]
            a = float(a)
            b = float(b)
            if a > b:
                raise ValueError(
                    f"deform_range start {a} must be <= end {b} (no wrap)."
                )
            # One-sided Gaussian tapers: flat 1 inside, falling outside.
            g_a = torch.where(
                theta >= a,
                torch.ones_like(theta),
                torch.exp(-((wrap(theta - a)) ** 2) / (2 * sig**2)),
            )
            g_b = torch.where(
                theta <= b,
                torch.ones_like(theta),
                torch.exp(-((wrap(theta - b)) ** 2) / (2 * sig**2)),
            )
            return (g_a * g_b)[:, None]
        c = float(self.deform_range)
        return torch.exp(-((wrap(theta - c)) ** 2) / (2 * sig**2))[:, None]

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
        model = get_model(data["model"])
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
            # Deform-range gate: the camera-position azimuth selects which
            # views deform at all; gated-out views keep the identity.
            j3d = j3d * self.view_mask(delta)[:, None, None, :]
        return j3d + torch.eye(3, device=self.device, dtype=j3d.dtype)

    def delta_jacobians(
        self, deltas: Float[Tensor, "M 3"]
    ) -> Float[Tensor, "M F 3 3"]:
        """Deformation jacobians evaluated at multiple view directions.

        Batched version of the ``delta`` branch of :meth:`jacobians_3d`,
        without the identity term.

        Args:
            deltas: View directions (mesh centroid minus camera location),
                (M, 3).

        Returns:
            out: Per-view deformation jacobians (M, F, 3, 3).
        """
        j3d = self.J_deform.from_cartesian(deltas)  # (F, M, 9)
        out = j3d.reshape(-1, deltas.shape[0], 3, 3).permute(1, 0, 2, 3)
        # Deform-range gate, matching the deformed() path so the
        # regularizer sees the same gated deformation.
        return out * self.view_mask(deltas)[:, None, None, None]

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
        out = m.copy()
        out.V = V_new.contiguous()
        return out

    def train(
        self,
        model: bool = False,
        centroids: bool = True,
        sigmas: bool = True,
        weights: bool = True,
        **kwargs,
    ):
        super().train(model=model, **kwargs)
        if not centroids:
            self.J_deform.centroids.requires_grad_(False)
        if not sigmas:
            self.J_deform.log_sigmas.requires_grad_(False)
        if not weights:
            self.J_deform.weights.requires_grad_(False)
