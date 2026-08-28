from __future__ import annotations
import torch

from math import tan, radians, exp
from typing import Tuple, Optional
from jaxtyping import Float
from torch import Tensor

from .math import RAD2DEG
from .quaternion import Quaternion


class CameraCoordinates:
    """
    Holds camera coordinates.
    """

    def __init__(
        self,
        origin: torch.Tensor = None,
        Q: Quaternion = None,
        radius: float = 1,
    ):
        self.origin = torch.tensor([0.0, 0.0, 0.0]) if origin is None else origin
        self.radius: float = radius
        self.Q = Quaternion.identity() if Q is None else Q
        self.up = torch.tensor([0, 0, 1])

    @property
    def device(self) -> torch.device:
        return self.Q.device

    @property
    def dtype(self) -> torch.dtype:
        return self.Q.dtype

    def to(self, *args, **kwargs) -> Camera:
        self.origin = self.origin.to(*args, **kwargs)
        self.Q = self.Q.to(*args, **kwargs)
        self.up = self.up.to(*args, **kwargs)
        return self

    def requires_grad_(self, mode: bool) -> Camera:
        self.origin.requires_grad_(mode)
        self.Q.requires_grad_(mode)

    def copy(self) -> "CameraCoordinates":
        """
        Return a copy of the object.
        """
        return CameraCoordinates(self.origin.clone(), self.Q.clone(), self.radius)


class Camera:
    """
    Holds camera parameters, and its coordinates object.
    """

    def __init__(
        self,
        H: int = 512,
        W: int = 512,
        F: int = 60,
        Zn: float = 0.1,
        Zf: float = 100.0,
        co: CameraCoordinates = None,
    ):
        self.F = F
        self.Zn = Zn
        self.Zf = Zf
        self.co = CameraCoordinates() if co is None else co

        self.set_window_size(H, W)

    @property
    def device(self) -> torch.device:
        return self.co.device

    @property
    def dtype(self) -> torch.dtype:
        return self.co.dtype

    def to(self, *args, **kwargs) -> Camera:
        self.co = self.co.to(*args, **kwargs)
        return self

    def requires_grad_(self, mode: bool):
        self.co.requires_grad_(mode)

    def set_window_size(self, H: int, W: int):
        """
        Sets the window size of the camera, and updates related variables.
        """
        assert H >= 1 and W >= 1, "Camera size cannot be smaller than 1."
        self.H = H
        self.W = W
        self.ratio = W / H
        self.Fx = W / (2.0 * tan(radians(self.F) / 2.0))
        self.Cx = W / 2.0
        self.Cy = H / 2.0

    @property
    def location(self) -> Float[Tensor, "3"]:
        co = self.co
        return (
            co.origin
            + co.Q.rotate_vector(
                torch.tensor([0, 0, -1], dtype=self.dtype, device=self.device)
            )
            * co.radius
        )

    @property
    def K(self) -> Float[Tensor, "3 3"]:
        """
        Returns the camera to image (3x3) matrix.
        """
        return torch.tensor(
            [[self.Fx, 0, self.Cx], [0, self.Fx, self.Cy], [0, 0, 1]],
            dtype=self.dtype,
            device=self.device,
        )

    @property
    def R(self) -> Float[Tensor, "3 3"]:
        """
        Returns camera's rotation matrix
        """
        return self.co.Q.R()

    @property
    def w2c(self) -> Float[Tensor, "4 4"]:
        """
        Returns the 4x4 world-to-camera transform matrix, fully differentiable.
        """
        R = self.R
        t = -R @ self.location  # (3,)
        Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (3,4)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device, dtype=R.dtype)
        mat = torch.cat([Rt, bottom], dim=0)  # (4,4)
        return mat

    @property
    def c2w(self) -> Float[Tensor, "4 4"]:
        """
        Returns a camera to world (4x4) projection matrix.
        """
        return torch.inverse(self.w2c)

    @property
    def Rt(self) -> Tuple[Float[Tensor, "3 3"], Float[Tensor, "3 1"]]:
        """
        Returns Rotation (R, 3x3) and translation(t, 3x1) matrices, in that order.
        """
        mat = self.w2c
        return mat[:3, :3], mat[:3, 3]

    def project_w2c(self, pts: Float[Tensor, "N 3"]) -> Float[Tensor, "N 3"]:
        """
        Projects points from world space to camera space.
        """
        R, t = self.Rt
        return (R @ pts.T + t[:, None]).T

    def project_c2i(
        self, pts: Float[Tensor, "N 3"]
    ) -> Tuple[Float[Tensor, "N 3"], Float[Tensor, "N"]]:
        """
        Project set of points in camera space to image space.
        Returns new coordinates and respective depths.
        """
        assert pts.shape[-1] == 3
        K = self.K()[:2, :]
        Z = pts[:, 2]
        return (K @ pts.T / Z.unsqueeze(0)).T, Z

    def project_w2i(self, pts: Float[Tensor, "N 3"]) -> Float[Tensor, "N 3"]:
        """
        Project set of points in cartesian space to image space.
        """
        c = self.project_w2c(pts)
        i, _ = self.project_c2i(c)
        return i

    def translate(self, vec: Float[Tensor, "3"]):
        """
        Move the camera along world axes.
        """
        assert vec.shape == self.co.origin.shape, (
            f"{vec.shape} != {self.co.origin.shape}"
        )
        self.co.origin += vec

    def look_at(self, vec: torch.Tensor):  # ToDO: verify this
        """
        Orient the camera so it looks at `target` from its current origin.
        """
        self.co = CameraCoordinates(
            origin=vec.clone(), Q=Quaternion.identity(self.device)
        )

    def rotate_from_image_space(self, dx: int, dy: int, deg: float):
        """
        Rotate the camera around its own origin based on image-space motion and a rotation angle.
        """
        R = self.R
        # Compute rotation axis in world space
        axis = -dx * R[1, :] + dy * R[0, :]
        angle = torch.tensor(RAD2DEG * deg, device=axis.device)
        self.co.Q *= Quaternion.from_axis_angle(axis, angle)

    def roll_from_image_space(self, deg: float):
        """
        Rotate the camera on its forward axis.
        """
        axis = self.R[2, :]
        angle = torch.tensor(RAD2DEG * deg, device=axis.device)
        rotation = Quaternion.from_axis_angle(axis, angle)
        self.co.Q *= rotation

    def translate_image_space(self, dx: int, dy: int, dist: float):
        R = self.R
        direction = -dx * R[0, :] - dy * R[1, :]
        self.translate(direction / direction.norm() * dist)

    def translate_depth(self, depth: float):
        self.co.radius *= exp(-depth * 0.1)

    def copy(self) -> Camera:
        return Camera(self.H, self.W, self.F, self.Zf, self.Zn, self.co.copy())

    def clone(self) -> Camera:
        return self.copy()

    @staticmethod
    def random_rot(
        H: int = 512,
        W: int = 512,
        require_grad: bool = False,
        origin: Optional[torch.Tensor] = None,
    ) -> Camera:
        if origin is None:
            origin = torch.tensor([0.0, 0.0, 0.0])
        cam = Camera(
            H=H,
            W=W,
            co=CameraCoordinates(origin=origin, Q=Quaternion.random()),
        )
        if require_grad:
            cam.requires_grad_(require_grad)
        return cam
