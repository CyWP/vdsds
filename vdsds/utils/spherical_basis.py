from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn


class SphericalGaussianBasis(nn.Module):
    """Basis of spherical Gaussians on SO(3) for view-dependent interpolation.

    With ``batched_basis=False`` (default) the Gaussian geometry (centroids,
    widths) is shared across the batch and only the weights vary per batch
    element: every instance interpolates the same set of bumps with its own
    weight vector. With ``batched_basis=True`` each batch element carries its
    own full function set (the pre-refactor storage layout).

    Attributes:
        centroids (nn.Parameter): (N, 3) or, when ``batched_basis``, (B, N, 3)
            centroid *directions* (arbitrary 3D vectors, normalized internally
            for evaluation).
        log_sigmas (nn.Parameter): (N,) or, when ``batched_basis``, (B, N) log
            of the falloff widths in dot-product space: the gaussian fall-off
            evaluated at a query direction with dot-product distance
            ``g = 1 - u @ m`` is ``exp(-g / (2 * sigma^2))`` with
            ``sigma = exp(log_sigmas)``.
        weights (nn.Parameter): (B, N, num_dims) values interpolated per
            function, one weight vector per batch element.

    Construction:
        SphericalGaussianBasis(num_funcs, num_dims, batch_size, ...) -> SphericalGaussianBasis

    Notes:
        - The falloff is a function of the dot product ``g = 1 - u @ m`` with
          the normalized centroid direction, NOT of the angular distance:
          this makes evaluation and, in particular, all gradients finite and
          well defined everywhere (poles, antipodes, exact-center queries,
          collapsed widths). Near coincidence ``1 - cos(theta) ~ theta^2 / 2``
          so dot-space widths relate to radian sigmas by
          ``sigma_dot = sigma_radian^2 / 2`` for the same
          ``exp(-theta^2 / (2 sigma_r^2))`` falloff (so the 1/e angular
          radius is ``sqrt(2) * sigma_r`` in both parametrizations).
        - Distant queries have softer tails than the equivalent
          radians-parameterized gaussian (dot form decays only to
          ``exp(-2 / (2 sigma_dot))`` at the antipode).
        - Default widths preserve the mass-covering init of the old
          radian parametrization:
          ``sigma_r = sqrt(2 * sigma_overlap / num_funcs)``, i.e.
          ``N * 2 * pi * sigma_r^2 = 4 * pi * sigma_overlap``.
        - With ``normalize=True`` the per-function influence is rescaled at
          every query point so the basis values sum to 1 (partition of
          unity): evaluation becomes a true weighted average of the
          weights, so equal weights yield a constant output field.
        - ``_basis`` takes plain ``(M, 3)`` query directions (no batch);
          with shared geometry the output is ``(M, N)`` and coverage
          analysis is ``basis.sum(-1) -> (M,)``.
    """

    def __init__(
        self,
        num_funcs: int,
        num_dims: int,
        batch_size: int,
        weights: Float[Tensor, "B N D"] | None = None,
        log_sigmas: Tensor | None = None,
        centroids: Tensor | None = None,
        init: str = "fibonacci",
        sigma_overlap: float = 2.0,
        normalize: bool = False,
        batched_basis: bool = False,
    ):
        """
        Args:
            num_funcs: Number of Gaussian basis functions N.
            num_dims: Number of interpolated dimensions D.
            batch_size: Number of batch elements B (per-instance weights).
            weights: Optional explicit weights (B, N, D), zeros otherwise.
            log_sigmas: Optional explicit log widths; (N,) shared or
                (B, N) when ``batched_basis``.
            centroids: Optional explicit centroid directions; (N, 3) shared
                or (B, N, 3) when ``batched_basis``. Arbitrary nonzero
                vectors (normalized internally).
            init: Centroid initialization, "fibonacci" or "random".
            sigma_overlap: Fraction of the SO(3) sphere surface covered by
                the total gaussian mass (see the class Notes for the exact
                relation to dot-space widths). Only used when `log_sigmas`
                is not provided.
            normalize: If True, rescale the per-function influence at every
                query point so the basis values sum to 1 (partition of
                unity / weighted-average interpolation).
            batched_basis: If True, keep a batch dimension on centroids and
                log widths — one full function set per batch element.
        """
        super().__init__()
        self.num_funcs = num_funcs
        self.num_dims = num_dims
        self.batch_size = batch_size
        self.normalize = normalize
        self.batched_basis = batched_basis
        self._geom_batch = (batch_size,) if batched_basis else ()

        if centroids is not None:
            assert centroids.shape == self._geom_batch + (num_funcs, 3)
        else:
            centroids = self._init_centroids(num_funcs, init)  # (N, 3)
            if batched_basis:
                centroids = centroids[None].expand(batch_size, -1, -1).clone()
        self.centroids = nn.Parameter(centroids)

        if log_sigmas is not None:
            assert log_sigmas.shape == self._geom_batch + (num_funcs,)
        else:
            theta = math.sqrt(2 * sigma_overlap / num_funcs)  # radian sigma
            # Matching exp(-theta^2 / (2 sigma^2)): 1 - cos(theta) ~ theta^2/2,
            # so the dot-space width with falloff exp(-g / (2 sigma_dot))
            # reproduces the radian falloff for sigma_dot = theta^2 / 2.
            sigma_dot = theta**2 / 2
            log_sigmas = torch.full((num_funcs,), math.log(sigma_dot))  # (N,)
            if batched_basis:
                log_sigmas = log_sigmas[None].expand(batch_size, -1).clone()
        self.log_sigmas = nn.Parameter(log_sigmas)

        if weights is not None:
            assert weights.shape == (batch_size, num_funcs, num_dims)
        else:
            weights = torch.zeros(batch_size, num_funcs, num_dims)
        self.weights = nn.Parameter(weights)

    @staticmethod
    def _init_centroids(N: int, init: str) -> Float[Tensor, "N 3"]:
        if init == "fibonacci":
            i = torch.arange(N, dtype=torch.float32) + 0.5
            z = 1 - i / N  # (N,)
            r = (1 - z * z).clamp(min=0.0).sqrt()
            golden = math.pi * (1 + math.sqrt(5))
            phi = (golden * i) % (2 * math.pi)  # (N,)
            return torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1)  # (N, 3)
        if init == "random":
            z = torch.rand(N) * 2 - 1  # (N,)
            phi = torch.rand(N) * 2 * math.pi
            r = (1 - z * z).clamp(min=0.0).sqrt()
            return torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1)
        raise ValueError(f"Unknown init: {init}")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        normalize: bool = False,
        batched_basis: bool | None = None,
    ) -> SphericalGaussianBasis:
        """Reconstruct an instance from its dict representation.

        Args:
            data: Dict with "weights" and optionally "log_sigmas"/"centroids",
                as produced by :meth:`to_dict`.
            normalize: Whether to rescale the basis to a partition of unity
                at every query point (configuration, not part of `to_dict`).
            batched_basis: Geometry layout override. If None (default), read
                from the dict; dicts saved before this feature have no such
                key and default to True (per-batch geometry).

        Note:
            Checkpoints in old formats (2D polar "centroids",
            radian-valued "sigmas") are not compatible with this class.

        Returns:
            out: SphericalGaussianBasis restored from the given tensors.
        """
        if batched_basis is None:
            batched_basis = bool(data.get("batched_basis", True))
        weights = data["weights"]
        batch_size, num_funcs, num_dims = weights.shape
        return cls(
            num_funcs=num_funcs,
            num_dims=num_dims,
            batch_size=batch_size,
            weights=weights,
            log_sigmas=data.get("log_sigmas"),
            centroids=data.get("centroids"),
            normalize=normalize,
            batched_basis=batched_basis,
        )

    def to_dict(self) -> dict[str, object]:
        """Serializes the basis into a plain dict of tensors.

        Tensors are cloned, detached and moved to cpu.

        Returns:
            out: Dict with "weights", "log_sigmas", "centroids" and the
                booleans "normalize" and "batched_basis".
        """
        return {
            "weights": self.weights.clone().detach().cpu(),
            "log_sigmas": self.log_sigmas.clone().detach().cpu(),
            "centroids": self.centroids.clone().detach().cpu(),
            "normalize": self.normalize,
            "batched_basis": self.batched_basis,
        }

    def copy(self) -> SphericalGaussianBasis:
        """Deep copy with cloned parameters.

        Returns:
            out: New SphericalGaussianBasis with cloned centroids, log
            widths and weights.
        """
        return SphericalGaussianBasis(
            num_funcs=self.num_funcs,
            num_dims=self.num_dims,
            batch_size=self.batch_size,
            weights=self.weights.clone(),
            log_sigmas=self.log_sigmas.clone(),
            centroids=self.centroids.clone(),
            normalize=self.normalize,
            batched_basis=self.batched_basis,
        )

    def smoothness_loss(self) -> Float[Tensor, ""]:
        """Penalize weight differences between overlapping Gaussians.

        Returns:
            out: Overlap-weighted mean of squared pairwise weight differences.
        """
        # Pairwise overlaps from the dot-product falloff. Diagonal pairs
        # (g = 0) evaluate to exp(0) = 1 and every gradient stays finite.
        if self.batched_basis:
            m = self._axis_dirs()  # (B, N, 3)
            g = (1.0 - (m[:, :, None] * m[:, None, :]).sum(-1)).clamp_min(0.0)  # (B, N, N)
            w = self.log_sigmas.exp().clamp_min(torch.finfo(m.dtype).tiny)  # (B, N)
            # Product of the two independent gaussian tails:
            # exp(-g/(2w_i)) * exp(-g/(2w_j)) = exp(-g*(w_i + w_j)/(2 w_i w_j))
            exponent = g * 0.5 * (1 / w[:, :, None] + 1 / w[:, None, :])
            overlap = torch.exp(-exponent)  # (B, N, N)
        else:
            m = self._axis_dirs()  # (N, 3)
            g = (1.0 - (m[:, None] * m[None, :]).sum(-1)).clamp_min(0.0)  # (N, N)
            w = self.log_sigmas.exp().clamp_min(torch.finfo(m.dtype).tiny)  # (N,)
            exponent = g * 0.5 * (1 / w[:, None] + 1 / w[None, :])
            overlap = torch.exp(-exponent)  # (N, N)

        diff2 = (self.weights.unsqueeze(1) - self.weights.unsqueeze(2)).pow(2).sum(-1)
        # (B, N, 1, D) - (B, 1, N, D) -> (B, N, N, D) -> (B, N, N)
        if self.batched_basis:
            out = (overlap * diff2).sum(dim=(1, 2)) / overlap.sum(dim=(1, 2)).clamp(
                min=1e-8
            )
        else:
            # Shared geometry: one overlap matrix, same denominator for all
            # batch elements.
            out = (overlap[None] * diff2).sum(dim=(1, 2)) / overlap.sum().clamp(
                min=1e-8
            )
        return out.mean()

    def _axis_dirs(self) -> Tensor:
        """Normalized centroid direction vectors.

        Returns:
            m: (N, 3) unit vectors of the given centroids, or (B, N, 3)
            when ``batched_basis``.
        """
        norm2 = (
            (self.centroids * self.centroids)
            .sum(-1, keepdim=True)
            .clamp_min(torch.finfo(self.centroids.dtype).tiny)
        )
        return self.centroids / norm2.sqrt()

    def _basis(self, u: Float[Tensor, "... M 3"]) -> Float[Tensor, "... M N"]:
        """Evaluates every Gaussian at every query direction.

        Args:
            u: Unit query directions (M, 3) or (B, M, 3) — arbitrary
                leading batch dims when the geometry is shared. With
                ``batched_basis``, (B, M, 3) uses per-face centroids and
                a plain (M, 3) is expanded across the batch.

        Returns:
            out: Raw (unnormalized) influences (..., M, N) — with
            ``batched_basis`` always (B, M, N).
        """
        m = self._axis_dirs()  # (N, 3) or (B, N, 3)
        # Widths are positive via log-space; clamp away from 0 so the
        # inverse stays finite. Multiply by the reciprocal instead of
        # dividing by tiny widths: the backward pass of a division
        # involves 1/denominator^2, which overflows to inf for tiny
        # widths and turns the gradient into NaN.
        tiny = torch.finfo(u.dtype).tiny
        if self.batched_basis:
            if u.ndim == 2:
                u = u[None].expand(self.batch_size, -1, -1)  # (M, 3) -> (B, M, 3)
            g = (1.0 - (u.unsqueeze(-2) * m.unsqueeze(1)).sum(-1)).clamp_min(
                0.0
            )  # (B, M, N)
            inv = 0.5 / self.log_sigmas.exp().clamp_min(tiny).unsqueeze(1)  # (B, 1, N)
        else:
            g = (1.0 - (u.unsqueeze(-2) * m).sum(-1)).clamp_min(0.0)  # (M, N) or (B, M, N)
            inv = 0.5 / self.log_sigmas.exp().clamp_min(tiny)  # (N,)
        return torch.exp(-g * inv)  # (M, N) or (B, M, N)

    def _from_unit_directions(
        self, u: Float[Tensor, "B M 3"] | Float[Tensor, "M 3"]
    ) -> Float[Tensor, "B M D"]:
        """Interpolates weights for unit query directions.

        Evaluates the spherical gaussians from the dot-product distance
        (1 - u @ m), which is smooth and has finite gradients everywhere,
        including at the poles, the antipodes and exactly on a centroid.

        Args:
            u: Unit query directions (M, 3) shared across the batch, or
                (B, M, 3) for per-batch-element directions.

        Returns:
            out: Interpolated values (B, M, D).
        """
        basis = self._basis(u)  # (M, N) or (B, M, N)
        if self.normalize:
            # Rescale to a partition of unity per query point. The clamp
            # guards against all-underflow (tiny widths, antipodal queries):
            # the row stays 0 and the output falls back to 0, not NaN.
            basis = basis / basis.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if basis.ndim == 2:
            out = torch.einsum(
                "mn,fnd->fmd", basis, self.weights
            )  # (M, N), (B, N, D) -> (B, M, D)
        else:
            out = torch.einsum(
                "bmn,bnd->bmd", basis, self.weights
            )  # (B, M, N), (B, N, D) -> (B, M, D)
        return out

    def from_polar(
        self, theta: Float[Tensor, "M ..."], phi: Float[Tensor, "M ..."]
    ) -> Float[Tensor, "B M D"]:
        """Interpolates weights for polar angles (pure-trig, gradient safe).

        Args:
            theta: Polar angles, scalar, (M,), or (B, M).
            phi: Azimuthal angles, same shape as theta.

        Returns:
            out: Interpolated values (B, M, D).
        """
        # () -> (1,)
        if theta.ndim == 0:
            theta = theta.reshape(1)
            phi = phi.reshape(1)
        if theta.ndim == 2:
            assert theta.shape[0] == phi.shape[0] == self.batch_size, (
                f"Expected batch {self.batch_size}, got {theta.shape[0]}"
            )

        u = torch.stack(
            (theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()), dim=-1
        )  # (M, 3) or (B, M, 3)
        return self._from_unit_directions(u)

    def forward(
        self,
        theta: Float[Tensor, "M ..."],
        phi: Float[Tensor, "M ..."] | None = None,
        cartesian_co: bool = False,
    ) -> Float[Tensor, "B M D"]:
        if cartesian_co:
            return self.from_cartesian(theta)
        assert phi is not None, "phi required for polar coordinates"
        return self.from_polar(theta, phi)

    def from_cartesian(self, vec: Float[Tensor, "... 3"]) -> Float[Tensor, "B M D"]:
        """Interpolates weights for cartesian query directions.

        Args:
            vec: Cartesian directions (3,), (M, 3), (B, 3), or (B, M, 3).

        Returns:
            out: Interpolated values (B, M, D).
        """
        # Normalize to (M, 3) or (B, M, 3)
        if vec.ndim == 1:
            vec = vec[None]  # (3,) -> (1, 3)
        elif vec.ndim == 2 and vec.shape[0] == self.batch_size:
            vec = vec.unsqueeze(1)  # (B, 3) -> (B, 1, 3)
        elif vec.ndim == 3 and vec.shape[0] != self.batch_size:
            vec = vec.expand(self.batch_size, -1, -1)

        # Unit query directions, clamped norm so poles and zero vectors
        # stay finite AND differentiable.
        r2 = (vec * vec).sum(-1, keepdim=True).clamp_min(torch.finfo(vec.dtype).tiny)
        u = vec / r2.sqrt()  # (M, 3) or (B, M, 3)
        return self._from_unit_directions(u)
