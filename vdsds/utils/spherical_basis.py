from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn


class SphericalGaussianBasis(nn.Module):
    """Basis of spherical Gaussians on SO(3) for view-dependent interpolation.

    Attributes:
        centroids (nn.Parameter): (B, N, 3) centroid *directions* (arbitrary
            3D vectors, normalized internally for evaluation).
        log_sigmas (nn.Parameter): (B, N) log of the falloff widths in
            dot-product space: the gaussian fall-off evaluated at a query
            direction with dot-product distance ``g = 1 - u @ m`` is
            ``exp(-g / (2 * sigma^2))`` with ``sigma = exp(log_sigmas)``.
        weights (nn.Parameter): (B, N, num_dims) values interpolated per function.

    Construction:
        SphericalGaussianBasis(num_funcs, num_dims, batch_size, ...) -> SphericalGaussianBasis

    Notes:
        - The falloff is a function of the dot product ``g = 1 - u @ m`` with
          the normalized centroid direction, NOT of the angular distance:
          this makes evaluation and, in particular, all gradients finite and
          well defined everywhere (poles, antipodes, exact-center queries,
          collapsed widths). Near coincidence ``1 - cos(theta) ~ theta^2 / 2``
          so dot-space widths relate to the previous radian sigmas by
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
    """

    def __init__(
        self,
        num_funcs: int,
        num_dims: int,
        batch_size: int,
        weights: Float[Tensor, "B N D"] | None = None,
        log_sigmas: Float[Tensor, "B N"] | None = None,
        centroids: Float[Tensor, "B N 3"] | None = None,
        init: str = "fibonacci",
        sigma_overlap: float = 2.0,
        normalize: bool = False,
    ):
        """
        Args:
            num_funcs: Number of Gaussian basis functions N.
            num_dims: Number of interpolated dimensions D.
            batch_size: Batch size B.
            weights: Optional explicit weights (B, N, D), zeros otherwise.
            log_sigmas: Optional explicit log widths (B, N) in dot-space.
            centroids: Optional explicit centroid directions (B, N, 3),
                arbitrary nonzero vectors (normalized internally).
            init: Centroid initialization, "fibonacci" or "random".
            sigma_overlap: Fraction of the SO(3) sphere surface covered by
                the total gaussian mass (see the class Notes for the exact
                relation to dot-space widths). Only used when `log_sigmas`
                is not provided.
            normalize: If True, rescale the per-function influence at every
                query point so the basis values sum to 1 (partition of
                unity / weighted-average interpolation).
        """
        super().__init__()
        self.num_funcs = num_funcs
        self.num_dims = num_dims
        self.batch_size = batch_size
        self.normalize = normalize

        if centroids is not None:
            assert centroids.shape == (batch_size, num_funcs, 3)
        else:
            centroids = self._init_centroids(batch_size, num_funcs, init)
        self.centroids = nn.Parameter(centroids)

        if log_sigmas is not None:
            assert log_sigmas.shape == (batch_size, num_funcs)
        else:
            theta = math.sqrt(2 * sigma_overlap / num_funcs)  # radian sigma
            # Matching exp(-theta^2 / (2 sigma^2)): 1 - cos(theta) ~ theta^2/2,
            # so the dot-space width with falloff exp(-g / (2 sigma_dot))
            # reproduces the old radian falloff for sigma_dot = theta^2 / 2.
            sigma_dot = theta**2 / 2
            log_sigmas = torch.full((batch_size, num_funcs), math.log(sigma_dot))
        self.log_sigmas = nn.Parameter(log_sigmas)

        if weights is not None:
            assert weights.shape == (batch_size, num_funcs, num_dims)
        else:
            weights = torch.zeros(batch_size, num_funcs, num_dims)
        self.weights = nn.Parameter(weights)

        self.register_buffer("ones_shape", torch.ones((self.batch_size, 1)))

    @staticmethod
    def _init_centroids(B: int, N: int, init: str) -> Float[Tensor, "B N 3"]:
        if init == "fibonacci":
            i = torch.arange(N, dtype=torch.float32) + 0.5
            z = 1 - i / N  # (N,)
            r = (1 - z * z).clamp(min=0.0).sqrt()
            golden = math.pi * (1 + math.sqrt(5))
            phi = (golden * i) % (2 * math.pi)  # (N,)
            centroids = torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1)  # (N, 3)
            return centroids[None].expand(B, -1, -1).clone()
        if init == "random":
            z = torch.rand(B, N) * 2 - 1  # (B, N)
            phi = torch.rand(B, N) * 2 * math.pi
            r = (1 - z * z).clamp(min=0.0).sqrt()
            return torch.stack([r * phi.cos(), r * phi.sin(), z], dim=-1)
        raise ValueError(f"Unknown init: {init}")

    @classmethod
    def from_dict(
        cls, data: dict[str, Tensor], normalize: bool = False
    ) -> SphericalGaussianBasis:
        """Reconstruct an instance from its dict representation.

        Args:
            data: Dict with "weights" and optionally "log_sigmas"/"centroids",
                as produced by :meth:`to_dict`.
            normalize: Whether to rescale the basis to a partition of unity
                at every query point (configuration, not part of `to_dict`).

        Note:
            Checkpoints in the old format (2D polar "centroids" and
            radian-valued "sigmas") are not compatible with this class.

        Returns:
            out: SphericalGaussianBasis restored from the given tensors.
        """
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
        )

    def to_dict(self) -> dict[str, Tensor]:
        """Serializes the basis into a plain dict of tensors.

        Tensors are cloned, detached and moved to cpu.

        Returns:
            out: Dict with "weights", "log_sigmas" and "centroids".
        """
        return {
            "weights": self.weights.clone().detach().cpu(),
            "log_sigmas": self.log_sigmas.clone().detach().cpu(),
            "centroids": self.centroids.clone().detach().cpu(),
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
        )

    def smoothness_loss(self) -> Float[Tensor, ""]:
        """Penalize weight differences between overlapping Gaussians.

        Returns:
            out: Overlap-weighted mean of squared pairwise weight differences.
        """
        # Pairwise overlaps from the dot-product falloff. Diagonal pairs
        # (g = 0) evaluate to exp(0) = 1 and every gradient stays finite.
        m = self._axis_dirs()  # (B, N, 3)
        g = (1.0 - (m.unsqueeze(2) * m.unsqueeze(1)).sum(-1)).clamp_min(
            0.0
        )  # (B, N, N)
        w = self.log_sigmas.exp().clamp_min(torch.finfo(m.dtype).tiny)  # (B, N)
        # Product of the two independent gaussian tails:
        # exp(-g/(2w_i)) * exp(-g/(2w_j)) = exp(-g*(w_i + w_j)/(2 w_i w_j))
        exponent = g * 0.5 * (1 / w.unsqueeze(1) + 1 / w.unsqueeze(2))
        overlap = torch.exp(-exponent)  # (B, N, N)

        diff2 = (self.weights.unsqueeze(1) - self.weights.unsqueeze(2)).pow(2).sum(-1)
        # (B, N, 1, D) - (B, 1, N, D) -> (B, N, N, D) -> (B, N, N)
        out = (overlap * diff2).sum(dim=(1, 2)) / overlap.sum(dim=(1, 2)).clamp(
            min=1e-8
        )
        return out.mean()

    def _axis_dirs(self) -> Float[Tensor, "B N 3"]:
        """Normalized centroid direction vectors.

        Returns:
            m: (B, N, 3) unit vectors of the given centroids.
        """
        norm2 = (
            (self.centroids * self.centroids)
            .sum(-1, keepdim=True)
            .clamp_min(torch.finfo(self.centroids.dtype).tiny)
        )
        return self.centroids / norm2.sqrt()

    def _from_unit_directions(
        self, u: Float[Tensor, "B M 3"]
    ) -> Float[Tensor, "B M D"]:
        """Interpolates weights for unit query directions.

        Evaluates the spherical gaussians from the dot-product distance
        (1 - u @ m), which is smooth and has finite gradients everywhere,
        including at the poles, the antipodes and exactly on a centroid.

        Args:
            u: unit query directions.

        Returns:
            out: Interpolated values (B, M, D).
        """
        m = self._axis_dirs().unsqueeze(1)  # (B, 1, N, 3)
        g = (1.0 - (u.unsqueeze(-2) * m).sum(-1)).clamp_min(0.0)  # (B, M, N)
        # Widths are positive via log-space; clamp away from 0 so the
        # inverse stays finite. Multiply by the reciprocal instead of
        # dividing by tiny widths: the backward pass of a division
        # involves 1/denominator^2, which overflows to inf for tiny
        # widths and turns the gradient into NaN.
        inv = 0.5 / self.log_sigmas.exp().unsqueeze(1).clamp_min(
            torch.finfo(u.dtype).tiny
        )  # (B, 1, N)
        basis = torch.exp(-g * inv)  # (B, M, N)
        if self.normalize:
            # Rescale to a partition of unity per query point. The clamp
            # guards against all-underflow (tiny widths, antipodal queries):
            # the row stays 0 and the output falls back to 0, not NaN.
            basis = basis / basis.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        out = basis @ self.weights  # (B, M, N) @ (B, N, D) -> (B, M, D)
        return out

    def from_polar(
        self, theta: Float[Tensor, "M ..."], phi: Float[Tensor, "M ..."]
    ) -> Float[Tensor, "B M D"]:
        """Interpolates weights for polar angles (pure-trig, gradient safe).

        Args:
            theta: Polar angles, scalar or (B, M).
            phi: Azimuthal angles, scalar or (B, M).

        Returns:
            out: Interpolated values (B, M, D).
        """
        # (M,) or () -> (B, M)
        if theta.ndim == 0:
            theta = self.ones_shape * theta
            phi = self.ones_shape * phi
        if theta.shape[0] != self.batch_size:
            theta = theta.expand(self.batch_size, -1)
            phi = phi.expand(self.batch_size, -1)
        assert theta.shape[0] == phi.shape[0] == self.batch_size, (
            f"Expected batch {self.batch_size}, got {theta.shape[0]}"
        )

        u = torch.stack(
            (theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()), dim=-1
        )  # (B, M, 3)
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
            vec: Cartesian directions (3,), (B, 3), or (B, M, 3).

        Returns:
            out: Interpolated values (B, M, D).
        """
        # Normalize to (B, M, 3)
        if vec.ndim == 1:
            vec = vec[None, None]  # (3,) -> (1, 1, 3)
        elif vec.ndim == 2:
            if vec.shape[0] == self.batch_size:
                vec = vec.unsqueeze(1)  # (B, 3) -> (B, 1, 3)
            else:
                vec = vec[None].expand(self.batch_size, -1, -1)  # (M, 3) -> (B, M, 3)
        elif vec.shape[0] != self.batch_size:
            vec = vec.expand(self.batch_size, -1, -1)

        # Unit query directions, clamped norm so poles and zero vectors
        # stay finite AND differentiable.
        r2 = (vec * vec).sum(-1, keepdim=True).clamp_min(torch.finfo(vec.dtype).tiny)
        u = vec / r2.sqrt()  # (B, M, 3)
        return self._from_unit_directions(u)
