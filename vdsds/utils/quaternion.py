from __future__ import annotations
import torch
from torch import Tensor
import random
from jaxtyping import Float
from typing import Tuple
from math import sin, cos, sqrt, pi
from .math import DEG2RAD, RAD2DEG


class Quaternion:
    """
    A unit quaternion class for 3D rotations, using PyTorch tensors.
    Stored as (w, x, y, z).
    Right multiplication is default: Q_new = Q_old * Q_transform
    """

    def __init__(self, q: Float[Tensor, "4"]):
        self.q = q / q.norm()

    @property
    def device(self) -> torch.device:
        return self.q.device

    @property
    def dtype(self) -> torch.dtype:
        return self.q.dtype

    def to(self, *args, **kwargs) -> Quaternion:
        self.q = self.q.to(*args, **kwargs)
        return self

    def requires_grad_(self, mode: bool):
        self.q.requires_grad_(mode)

    @classmethod
    def from_axis_angle(cls, axis: Float[Tensor, "3"], angle: float) -> Quaternion:
        axis = axis / axis.norm()
        half = angle * 0.5
        w = torch.cos(half)
        xyz = axis * torch.sin(half)
        q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
        return cls(q)

    @classmethod
    def random(cls) -> Quaternion:
        # Ken Shoemake implementation, Uniform on SO3
        r1, r2, r3 = random.random(), random.random(), random.random()
        w = sqrt(1 - r1) * sin(r2 * 2 * pi)
        x = sqrt(1 - r1) * cos(r2 * 2 * pi)
        y = sqrt(r1) * sin(r3 * 2 * pi)
        z = sqrt(r1) * cos(r3 * 2 * pi)
        return cls(torch.tensor([w, x, y, z]))

    @classmethod
    def from_euler(cls, x: float, y: float, z: float) -> Quaternion:
        """
        Create quaternion from Euler angles (radians).
        x = roll, y = pitch, z = yaw
        """
        x *= DEG2RAD
        y *= DEG2RAD
        z *= DEG2RAD
        cx, cy, cz = cos(x / 2), cos(y / 2), cos(z / 2)
        sx, sy, sz = sin(x / 2), sin(y / 2), sin(z / 2)

        w = cx * cy * cz + sx * sy * sz
        x = sx * cy * cz - cx * sy * sz
        y = cx * sy * cz + sx * cy * sz
        z = cx * cy * sz - sx * sy * cz
        q = torch.tensor([w, x, y, z])
        return cls(q)

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_matrix(cls, R: Float[Tensor, "3 3"]) -> Quaternion:
        m = R
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0.0:
            s = torch.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        return cls(torch.cat([w, x, y, z], dim=0))

    def normalize(self):
        self.q = self.q / self.q.norm()

    def conjugate(self) -> Quaternion:
        # Do not modify self.q in-place.
        # Build a new tensor for the conjugate and return a new Quaternion.
        w, x, y, z = self.q.unbind(-1)
        q_conj = torch.stack([w, -x, -y, -z], dim=-1)
        return Quaternion(q_conj)

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Right multiplication: self * other"""
        w1, x1, y1, z1 = self.q.unbind(-1)
        w2, x2, y2, z2 = other.q.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(torch.stack([w, x, y, z], dim=0))

    @staticmethod
    def tensor_rotation(
        q1: Float[Tensor, "N 4"], q2: Float[Tensor, "N 4"]
    ) -> Float[Tensor, "N 4"]:
        """
        Batched quaternion multiplication: q_new = q1 * q2
        Supports shape (B, 4) for both inputs.
        Fully differentiable.
        """
        assert q1.shape == q2.shape
        assert q1.shape[-1] == 4

        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def vector_alignment(
        v_src: Float[Tensor, "N 3"], v_dst: Float[Tensor, "N 3"], eps: float = 1e-8
    ) -> Float[Tensor, "N 4"]:
        v_src = v_src / (v_src.norm(dim=1, keepdim=True) + eps)
        v_dst = v_dst / (v_dst.norm(dim=1, keepdim=True) + eps)
        cross = torch.cross(v_src, v_dst, dim=1)
        dot = (v_src * v_dst).sum(dim=1, keepdim=True)

        quat = torch.cat([1.0 + dot + eps, cross], dim=1)
        quat = quat / (quat.norm(dim=1, keepdim=True) + eps)

        return quat

    @staticmethod
    def axis_rotations(
        axes: Float[Tensor, "N 3"], angles: Float[Tensor, "N"], eps: float = 1e-8
    ) -> Float[Tensor, "N 4"]:

        if angles.dim() == 1:
            angles = angles.unsqueeze(1)

        axes = axes / (axes.norm(dim=1, keepdim=True) + eps)

        half = 0.5 * angles
        sin_half = torch.sin(half)
        cos_half = torch.cos(half)

        quat = torch.cat([cos_half, sin_half * axes], dim=1)
        return quat

    def __imul__(self, other: Quaternion) -> Quaternion:
        result = self * other
        self.q = result.q
        return self

    def apply_rotation_to(self, other: Float[Tensor, "N 4"]) -> torch.Tensor:
        w1, x1, y1, z1 = self.q.unbind()
        w2, x2, y2, z2 = other.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    def rotate_vector(self, v: Float[Tensor, "N 3"]) -> torch.Tensor:
        zero = torch.tensor(0.0, device=v.device)
        qv_tensor = torch.cat([zero.unsqueeze(-1), v], dim=-1)  # (...,4)

        # Wrap into Quaternion (constructor should be gradient-safe)
        qv = Quaternion(qv_tensor)

        res = self.conjugate() * qv * self
        # Return vector part; keep batch dims
        return res.q[1:]

    def R(self) -> Float[Tensor, "3 3"]:
        """
        Returns 3x3 rotation matrix for left-handed coordinate system.
        Differentiable w.r.t. self.q
        """
        w, x, y, z = self.q

        return torch.stack(
            [
                torch.stack(
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

    def euler(self) -> Tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        w, x, y, z = self.q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = torch.sign(sinp) * (pi / 2)  # clamp at ±90°
        else:
            pitch = torch.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG

    @property
    def as_tuple(self):
        return tuple(self.q.tolist())

    @property
    def euler_tuple(self):
        r, p, y = self.euler()
        return (r.item(), p.item(), y.item())

    def copy(self) -> Quaternion:
        return Quaternion(self.q.clone())

    def clone(self) -> Quaternion:
        return self.copy()

    def __repr__(self) -> str:
        return f"Quaternion({self.q.tolist()})"
