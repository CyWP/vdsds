from __future__ import annotations
import torch
import torch.nn as nn
import math

from jaxtyping import Float
from torch import Tensor
from typing import Optional


class SphericalHarmonic(nn.Module):
    def __init__(
        self, degree: int, num_dims: int, batch_size: int, weights: Optional[Tensor[Float "B W"]] = None
    ):
        super().__init__()
        self.degree = degree
        self.num_coeffs = (degree + 1) ** 2
        self.num_dims = num_dims
        self.batch_size = batch_size

        # Each batch element has its own weights: [B, num_dims, num_coeffs]
        if weights is not None:
            assert weights.shape == (self.batch_size, self.num_dims, self.num_coeffs)
        self.weights = nn.Parameter(
            (
                weights
                if weights is not None
                else torch.randn(batch_size, num_dims, self.num_coeffs) * 0.001
            ),
        )
        self.register_buffer("ones_shape", torch.ones((self.batch_size, 1)))

    def copy(self) -> SphericalHarmonic:
        return SphericalHarmonic(
            degree=self.degree,
            num_dims=self.num_dims,
            batch_size=self.batch_size,
            weights=self.weights.clone(),
        )

    def _legendre_all(self, lmax: int, x: Float[Tensor, "B N"]) -> Float[Tensor, "B W"]:
        """
        Precompute Legendre polynomials
        """
        B, N = x.shape
        P = {0: torch.ones(B, N, 1, device=x.device, dtype=x.dtype)}  # P_0^0 = 1
        if lmax == 0:
            return P

        # P_1^0 = x, P_1^1 = -sqrt(1-x^2)
        P[1] = torch.stack([x, -torch.sqrt(torch.clamp(1 - x * x, min=0.0))], dim=-1)

        for l in range(2, lmax + 1):
            arr = []
            for m in range(l + 1):
                if m == l:
                    val = (
                        -(2 * l - 1)
                        * torch.sqrt(torch.clamp(1 - x * x, min=0.0))
                        * P[l - 1][..., m - 1]
                    )
                elif m == l - 1:
                    val = (2 * l - 1) * x * P[l - 1][..., m]
                else:
                    val = (
                        (2 * l - 1) * x * P[l - 1][..., m]
                        - (l + m - 1) * P[l - 2][..., m]
                    ) / (l - m)
                arr.append(val)
            P[l] = torch.stack(arr, dim=-1)
        return P

    def from_polar(self, theta: Float[Tensor, "N ..."], phi: Float[Tensor, "N ..."]) -> Float[Tensor, "B N Ndims"]:
        # theta, phi: [B, N]
        if len(theta.shape) == 0:
            theta = self.ones_shape * theta
            phi = self.ones_shape * phi
        if theta.shape[0] != self.batch_size:
            # Expand to batch size if needed
            theta = theta.expand(self.batch_size, -1)
            phi = phi.expand(self.batch_size, -1)
        assert theta.shape[0] == phi.shape[0] == self.batch_size, (
            f"Expected batch {self.batch_size}, got {theta.shape[0]}"
        )

        x = torch.cos(theta)  # [B,N]
        P = self._legendre_all(self.degree, x)

        basis = []
        for l in range(self.degree + 1):
            for m in range(-l, l + 1):
                m_abs = abs(m)
                norm = math.sqrt(
                    ((2 * l + 1) / (4 * math.pi))
                    * math.factorial(l - m_abs)
                    / math.factorial(l + m_abs)
                )
                leg = P[l][..., m_abs]  # [B,N]
                if m > 0:
                    basis.append(math.sqrt(2) * norm * leg * torch.cos(m * phi))
                elif m < 0:
                    basis.append(math.sqrt(2) * norm * leg * torch.sin(-m * phi))
                else:
                    basis.append(norm * leg)

        # Collect basis functions: [B,N,num_coeffs]
        Y = torch.stack(basis, dim=-1)

        # weights: [B,num_dims,num_coeffs] -> [B,1,num_dims,num_coeffs]
        W = self.weights.unsqueeze(1)

        # expand Y for broadcast: [B,N,1,num_coeffs]
        Y = Y.unsqueeze(2)

        # multiply and sum coeff dim
        out = (Y * W).sum(dim=-1)  # [B,N,num_dims]
        return out.squeeze(dim=1)

    def laplace_beltrami(self) -> Float[Tensor, "B N"]:
        out = torch.empty_like(self.weights)
        idx = 0
        for l in range(self.degree + 1):
            n_m = 2 * l + 1
            out[..., idx : idx + n_m] = (
                -l * (l + 1) * self.weights[..., idx : idx + n_m]
            )
            idx += n_m
        return out

    def smoothness_loss(self) -> Float[Tensor, ""]:
        return (self.weights * -self.laplace_beltrami()).mean()

    def from_cartesian(self, vec: Float[Tensor, "... 3"]) -> Float[Tensor, "B N Ndims"]:
        # vec: [B,3] or [3]
        if vec.ndim == 1:
            vec = vec.unsqueeze(0).expand(self.batch_size, -1)  # [B,3]
        elif vec.shape[0] != self.batch_size:
            vec = vec.expand(self.batch_size, -1)

        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        r = vec.norm(dim=1).clamp(min=1e-8)  # avoid division by zero

        theta = torch.acos(z / r)  # polar angle from Z-axis
        phi = torch.atan2(y, x)  # azimuthal angle in XY-plane

        # Handle poles: if theta is near 0 or pi, phi is undefined -> set to 0
        eps = 1e-6
        phi = torch.where(
            (theta < eps) | (theta > math.pi - eps), torch.zeros_like(phi), phi
        )

        theta = theta.unsqueeze(1)
        phi = phi.unsqueeze(1)

        return self.from_polar(theta, phi)
