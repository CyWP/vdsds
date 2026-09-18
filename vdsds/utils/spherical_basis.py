from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn


class SphericalGaussianBasis(nn.Module):
    """Basis of spherical Gaussians on the SO(3) sphere for view-dependent interpolation.

    Attributes:
        centroids (nn.Parameter): (B, N, 2) polar coordinates (theta, phi) of each Gaussian.
        sigmas (nn.Parameter): (B, N) angular falloff (in radians) of each Gaussian.
        weights (nn.Parameter): (B, N, num_dims) values interpolated per function.

    Construction:
        SphericalGaussianBasis(num_funcs, num_dims, batch_size, ...) -> SphericalGaussianBasis

    Notes:
        - Default sigmas satisfy sum_i 2*pi*sigma_i^2 = 4*pi*sigma_overlap, i.e. the
          total Gaussian mass covers a sigma_overlap fraction of the SO(3) sphere surface.
        - Evaluation is pole-safe via the haversine angular distance.
    """

    def __init__(
        self,
        num_funcs: int,
        num_dims: int,
        batch_size: int,
        weights: Float[Tensor, "B N D"] | None = None,
        sigmas: Float[Tensor, "B N"] | None = None,
        centroids: Float[Tensor, "B N 2"] | None = None,
        init: str = "fibonacci",
        sigma_overlap: float = 0.5,
    ):
        """
        Args:
            num_funcs: Number of Gaussian basis functions N.
            num_dims: Number of interpolated dimensions D.
            batch_size: Batch size B.
            weights: Optional explicit weights (B, N, D), zeros otherwise.
            sigmas: Optional explicit sigmas (B, N).
            centroids: Optional explicit centroids (B, N, 2).
            init: Centroid initialization, "fibonacci" or "random".
            sigma_overlap: Fraction of the SO(3) sphere surface covered by the total
                Gaussian mass, sum_i 2*pi*sigma_i^2 = 4*pi*sigma_overlap.
        """
        super().__init__()
        self.num_funcs = num_funcs
        self.num_dims = num_dims
        self.batch_size = batch_size

        if centroids is not None:
            assert centroids.shape == (batch_size, num_funcs, 2)
        else:
            centroids = self._init_centroids(batch_size, num_funcs, init)
        self.centroids = nn.Parameter(centroids)

        if sigmas is not None:
            assert sigmas.shape == (batch_size, num_funcs)
        else:
            # sum_i 2*pi*sigma_i^2 = 4*pi*sigma_overlap  ->  sigma_i = sqrt(2*sigma_overlap/N)
            sigma = math.sqrt(2 * sigma_overlap / num_funcs)
            sigmas = torch.full((batch_size, num_funcs), sigma)
        self.sigmas = nn.Parameter(sigmas)

        if weights is not None:
            assert weights.shape == (batch_size, num_funcs, num_dims)
        else:
            weights = torch.zeros(batch_size, num_funcs, num_dims)
        self.weights = nn.Parameter(weights)

        self.register_buffer("ones_shape", torch.ones((self.batch_size, 1)))

    @staticmethod
    def _init_centroids(B: int, N: int, init: str) -> Float[Tensor, "B N 2"]:
        if init == "fibonacci":
            i = torch.arange(N, dtype=torch.float32) + 0.5
            theta = torch.acos(1 - i / N)  # (N,)
            golden = math.pi * (1 + math.sqrt(5))
            phi = (golden * torch.arange(N, dtype=torch.float32)) % (
                2 * math.pi
            )  # (N,)
            centroids = torch.stack([theta, phi], dim=-1)  # (N, 2)
            return centroids[None].expand(B, -1, -1).clone()
        if init == "random":
            z = torch.rand(B, N) * 2 - 1  # (B, N)
            theta = torch.acos(z)
            phi = torch.rand(B, N) * 2 * math.pi
            return torch.stack([theta, phi], dim=-1)
        raise ValueError(f"Unknown init: {init}")

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Tensor]) -> SphericalGaussianBasis:
        """Reconstruct an instance from its parameter tensors.

        Args:
            state_dict: Dict with "weights" and optionally "sigmas"/"centroids".

        Returns:
            out: SphericalGaussianBasis restored from the given tensors.
        """
        weights = state_dict["weights"]
        batch_size, num_funcs, num_dims = weights.shape
        return cls(
            num_funcs=num_funcs,
            num_dims=num_dims,
            batch_size=batch_size,
            weights=weights,
            sigmas=state_dict.get("sigmas"),
            centroids=state_dict.get("centroids"),
        )

    def copy(self) -> SphericalGaussianBasis:
        """Deep copy with cloned parameters.

        Returns:
            out: New SphericalGaussianBasis with cloned centroids, sigmas, weights.
        """
        return SphericalGaussianBasis(
            num_funcs=self.num_funcs,
            num_dims=self.num_dims,
            batch_size=self.batch_size,
            weights=self.weights.clone(),
            sigmas=self.sigmas.clone(),
            centroids=self.centroids.clone(),
        )

    def smoothness_loss(self) -> Float[Tensor, ""]:
        """Penalize weight differences between overlapping Gaussians.

        Returns:
            out: Overlap-weighted mean of squared pairwise weight differences.
        """
        th0 = self.centroids[..., 0].unsqueeze(-1)  # (B, N, 1)
        ph0 = self.centroids[..., 1].unsqueeze(-1)  # (B, N, 1)

        cos_d = torch.cos(th0) * torch.cos(th0.transpose(1, 2)) + torch.sin(
            th0
        ) * torch.sin(th0.transpose(1, 2)) * torch.cos(ph0 - ph0.transpose(1, 2))
        hav = ((1 - cos_d) / 2).clamp(0, 1)
        d2 = 2 * torch.asin(hav.sqrt()) ** 2  # (B, N, N)

        # Gaussian overlap proxy between all centroid pairs
        s2 = self.sigmas.unsqueeze(1) ** 2 + self.sigmas.unsqueeze(2) ** 2  # (B, N, N)
        overlap = torch.exp(-d2 / s2)  # (B, N, N)

        diff2 = (self.weights.unsqueeze(1) - self.weights.unsqueeze(2)).pow(2).sum(-1)
        # (B, N, 1, D) - (B, 1, N, D) -> (B, N, N, D) -> (B, N, N)
        out = (overlap * diff2).sum(dim=(1, 2)) / overlap.sum(dim=(1, 2)).clamp(
            min=1e-8
        )
        return out.mean()

    def from_polar(
        self, theta: Float[Tensor, "M ..."], phi: Float[Tensor, "M ..."]
    ) -> Float[Tensor, "B M D"]:
        """
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

        th = theta.unsqueeze(-1)  # (B, M, 1)
        ph = phi.unsqueeze(-1)  # (B, M, 1)
        th0 = self.centroids[..., 0].unsqueeze(1)  # (B, 1, N)
        ph0 = self.centroids[..., 1].unsqueeze(1)  # (B, 1, N)

        # Haversine angular distance between query and centroids
        cos_d = torch.cos(th) * torch.cos(th0) + torch.sin(th) * torch.sin(
            th0
        ) * torch.cos(ph - ph0)  # (B, M, N)
        hav = ((1 - cos_d) / 2).clamp(0, 1)
        d2 = 2 * torch.asin(hav.sqrt()) ** 2  # (B, M, N)

        basis = torch.exp(-d2 / (2 * self.sigmas.unsqueeze(1) ** 2))  # (B, M, N)
        out = basis @ self.weights  # (B, M, N) @ (B, N, D) -> (B, M, D)
        return out

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
        """
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

        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
        r = vec.norm(dim=-1).clamp(min=1e-8)  # (B, M)

        theta = torch.acos(z / r)  # polar angle from Z-axis, (B, M)
        phi = torch.atan2(y, x)  # azimuthal angle in XY-plane, (B, M)

        # Poles: phi undefined, but haversine evaluation is independent of phi there
        return self.from_polar(theta, phi)
