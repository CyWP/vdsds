from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..utils.img import ImgUtils, Splimage
from ..utils.quaternion import Quaternion
from .camera import Camera, CameraCoordinates


def camera_to_dict(camera: Camera) -> dict:
    """Serialize a Camera into plain JSON-able types.

    Args:
        camera: Camera to serialize.

    Returns:
        out: Dict with H, W, F, Zn, Zf and the coordinates (origin,
            quaternion as (w, x, y, z), radius).
    """
    co = camera.co
    return {
        "H": camera.H,
        "W": camera.W,
        "F": camera.F,
        "Zn": camera.Zn,
        "Zf": camera.Zf,
        "co": {
            "origin": [float(v) for v in co.origin.tolist()],
            "Q": [float(v) for v in co.Q.q.tolist()],
            "radius": float(co.radius),
        },
    }


def camera_from_dict(data: dict) -> Camera:
    """Rebuild a Camera from :func:`camera_to_dict` output.

    Args:
        data: Serialized camera dict.

    Returns:
        out: Camera with cloned tensors on CPU.
    """
    co_data = data["co"]
    return Camera(
        H=int(data["H"]),
        W=int(data["W"]),
        F=data["F"],
        Zn=data["Zn"],
        Zf=data["Zf"],
        co=CameraCoordinates(
            origin=torch.tensor(co_data["origin"], dtype=torch.float32),
            Q=Quaternion(torch.tensor(co_data["Q"], dtype=torch.float32)),
            radius=float(co_data["radius"]),
        ),
    )


class Overlay:
    """Splimage-backed viewport overlay with live manipulable placement.

    The image is always stored as a :class:`Splimage` (RGBA BCHW tensor in
    [0, 1]); its source path is kept so the overlay state can be
    serialized and reconstructed later.
    """

    def __init__(
        self,
        img: Splimage | str,
        scale: float = 1.0,
        x: int = 0,
        y: int = 0,
        rot: float = 0.0,
        alpha: float = 0.35,
    ) -> None:
        """
        Args:
            img: Splimage or path-like to an image file (loaded via
                PIL through Splimage, forced RGBA).
            scale: Scale factor on top of the scale-to-fit viewport fit.
            x, y: Center offset in pixels from the viewport center.
            rot: Rotation in degrees (accumulated by the user).
            alpha: Compositing opacity of the overlay over renders.
        """
        self.path: str | None = None
        if isinstance(img, Splimage):
            self.img = img
            self.path = getattr(img, "path", None)
        else:
            if not isinstance(img, (str, os.PathLike)):
                self.img = Splimage(img)
            else:
                self.path = str(img)
                self.img = Splimage(self.path)
        self.scale = scale
        self.x = x
        self.y = y
        self.rot = rot
        self.alpha = alpha

    # LIVE MANIPULATION

    def translate(self, x: int, y: int) -> Overlay:
        """Shift the overlay by pixel deltas."""
        self.x += x
        self.y += y
        return self

    def rotate(self, deg: float) -> Overlay:
        """Rotate the overlay around its center."""
        self.rot += deg
        return self

    def scale_by(self, factor: float) -> Overlay:
        """Multiply the overlay's scale factor."""
        self.scale *= factor
        return self

    # RENDERING

    @property
    def device(self) -> torch.device:
        return self.img.device

    def image(self, H: int, W: int) -> Float[Tensor, "1 4 H W"]:
        """Return the overlay placed on an RGBA canvas of size (H, W).

        The image is padded to its diagonal square (so rotation never
        crops), rotated, scaled to fit the viewport (times ``scale``),
        translated by (x, y) and edge-clipped at the canvas boundaries.

        Args:
            H: Canvas height.
            W: Canvas width.

        Returns:
            out: Canvas (1, 4, H, W) in [0, 1] on the Splimage's device.
        """
        img = self.img.image()
        _, _, h, w = img.shape
        side = math.ceil(math.sqrt(h * h + w * w))
        pad_t = (side - h) // 2
        pad_b = side - h - pad_t
        pad_l = (side - w) // 2
        pad_r = side - w - pad_l
        rot_img = F.pad(img, (pad_l, pad_r, pad_t, pad_b))
        if self.rot:
            rot_img = self._rotate(rot_img)
        r_h, r_w = rot_img.shape[-2:]
        scale = self.scale * min(H / r_h, W / r_w)
        s_h = max(1, int(scale * r_h))
        s_w = max(1, int(scale * r_w))
        s_img = ImgUtils.resize(rot_img, s_h, s_w)
        canvas = torch.zeros((1, 4, H, W), device=s_img.device, dtype=s_img.dtype)
        Xmin = (W - s_w) // 2 + self.x
        Xmax = (W + s_w) // 2 + self.x
        Ymin = (H - s_h) // 2 + self.y
        Ymax = (H + s_h) // 2 + self.y
        cropped = s_img[
            :,
            :,
            max(0, -Ymin) : max(0, min(s_h, s_h - Ymax + H)),
            max(0, -Xmin) : max(0, min(s_w, s_w - Xmax + W)),
        ]
        canvas[
            :,
            :,
            max(0, Ymin) : min(H, Ymax),
            max(0, Xmin) : min(W, Xmax),
        ] = cropped
        return canvas

    def _rotate(self, img: Float[Tensor, "B C S S"]) -> Float[Tensor, "B C S S"]:
        """Rotate an image by ``rot`` degrees around its center."""
        angle = math.radians(self.rot)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=img.dtype,
            device=img.device,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        return F.grid_sample(img, grid, align_corners=False, padding_mode="zeros")

    def overlay(self, render: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        """Composite the overlay over a render.

        Args:
            render: Render batch (B, C, H, W), C in (3, 4), single-image
                viewport renders only (B == 1).

        Returns:
            out: Composited render; the original alpha channel (if any)
                is kept as-is behind the composited RGB.
        """
        B, C, H, W = render.shape
        assert B == 1, "Overlays composite over viewport renders (B == 1)."
        ov = self.image(H, W).to(render.device)
        o_rgb = ov[:, :3]
        o_alpha = ov[:, 3:4] * self.alpha
        out_rgb = o_alpha * o_rgb + (1 - o_alpha) * render[:, :3]
        if C == 4:
            return torch.cat([out_rgb, render[:, 3:4]], dim=1)
        return out_rgb

    # SERIALIZATION

    def to_dict(self) -> dict:
        """Serialize to JSON-able types.

        Raises:
            ValueError: If the Splimage was constructed without a source
                path and cannot be serialized.
        """
        if self.path is None:
            raise ValueError(
                "Overlay was constructed from a Splimage without a source "
                "path and cannot be serialized."
            )
        return {
            "img": self.path,
            "scale": float(self.scale),
            "x": int(self.x),
            "y": int(self.y),
            "rot": float(self.rot),
            "alpha": float(self.alpha),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Overlay:
        """Rebuild an Overlay from :meth:`to_dict` output."""
        return cls(
            img=Splimage(data["img"]),
            scale=data.get("scale", 1.0),
            x=data.get("x", 0),
            y=data.get("y", 0),
            rot=data.get("rot", 0.0),
            alpha=data.get("alpha", 0.35),
        )

    def copy(self) -> Overlay:
        """Deep copy (Splimage tensors are cloned)."""
        new = Overlay.__new__(Overlay)
        new.path = self.path
        new.img = self.img.copy() if isinstance(self.img, Splimage) else None
        new.scale = self.scale
        new.x = self.x
        new.y = self.y
        new.rot = self.rot
        new.alpha = self.alpha
        return new


class ViewState:
    """Everything about the current view: camera plus optional overlay.

    This is the snapshot unit captured into constraints.
    """

    def __init__(self, camera: Camera, overlay: Overlay | None = None) -> None:
        self.camera = camera
        self.overlay = overlay

    def to_dict(self) -> dict:
        """Serialize to JSON-able types."""
        return {
            "camera": camera_to_dict(self.camera),
            "overlay": self.overlay.to_dict() if self.overlay is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ViewState:
        """Rebuild a ViewState from :meth:`to_dict` output."""
        overlay = data.get("overlay", None)
        return cls(
            camera=camera_from_dict(data["camera"]),
            overlay=Overlay.from_dict(overlay) if overlay is not None else None,
        )

    def copy(self) -> ViewState:
        """Deep copy (camera and overlay tensors are cloned)."""
        return ViewState(
            camera=self.camera.copy(),
            overlay=self.overlay.copy() if self.overlay is not None else None,
        )
