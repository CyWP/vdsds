from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from ..rendering.camera import Camera, CameraCoordinates
from ..rendering.light import LightSource
from ..rendering.renderer import Renderer
from ..representations.base import Model
from ..utils.config import Config, Dict
from ..utils.img import ImgUtils, Splimage
from ..utils.quaternion import Quaternion
from ..utils.video import write_video
from .base import ViewableScript


class OrbitFrames(ViewableScript):
    """
    Loads a model (or deformation) without opening the viewer, and renders
    frames of a camera orbiting the model. The camera rotates using a
    :class:`Quaternion` at every frame, and each frame is written as an image
    in the output directory. The orbit can be horizontal, vertical, or both.

    Optionally, one or more **static** views are rendered alongside every orbit
    frame. Static views evaluate the deformation for the *orbit* camera's
    pose, so as the orbit moves they show the resulting deformation of the
    geometry from fixed viewpoints. The orbit and all static views are
    concatenated side by side (orbit first) into a single video.
    """

    _views: ClassVar[dict[str, tuple[Float[Tensor, "3"], float]]] = {
        # 'front' is the default camera view (orbit start). 'side' rotates
        # 90 degrees about the vertical from it (head-on, upright); 'back'
        # rotates 180 degrees from it. 'top'/'bottom' are bird's-eye views.
        "front": (CameraCoordinates._up, torch.pi / 2),
        "back": (CameraCoordinates._up, -torch.pi / 2),
        "side": (torch.tensor([1.0, 0.0, 0.0]), 0.0),
        "top": (torch.tensor([1.0, 0.0, 0.0]), torch.pi / 2),
        "bottom": (torch.tensor([1.0, 0.0, 0.0]), -torch.pi / 2),
    }

    _default_config_overrides: ClassVar[Config] = Config(
        {
            "window": {
                "view": False,
                "bg_color": [1.0, 1.0, 1.0],  # Color or None
            },
            "path": {
                "model": None,
                "out_dir": "./recordings",
                "file_name": "orbit",
            },
            "camera": {
                "H": 512,
                "W": 512,
                "F": 60,
                "radius": 1.5,  # None: auto from model bounding sphere.
            },
            "static_view": [  # Static views, rendered after the orbit and side
                # by side with it. Each entry: {"view": <preset>, "camera":
                # {H, W, F, radius}}; "camera" is optional.
                {"view": "top", "camera": {"H": 512, "W": 512, "F": 60}},
                {"view": "front", "camera": {"H": 512, "W": 512, "F": 60}},
                {"view": "side", "camera": {"H": 512, "W": 512, "F": 60}},
            ],
            "lighting": {
                "alignment": "up",  # Options: 'camera', 'up', or [x, y, z].
                "strength": 1.5,
            },
            "rotation": {
                "mode": "horizontal",  # Options: 'horizontal', 'vertical', 'both'.
                "frames": 240,
            },
        }
    )

    def run(self):
        cfg = self.config
        out_dir = Path(cfg.path.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.renderer = self.get_renderer()
        video_path = out_dir / f"{cfg.path.file_name}.mp4"
        write_video(self.orbit_all(), path=video_path)

    def orbit_all(self) -> Iterator[Splimage]:
        device = self.device
        cfg = self.config
        mode = cfg.rotation.mode
        if mode not in ("horizontal", "vertical", "both"):
            raise ValueError(f"Invalid orbit mode: '{mode}'.")
        rot_frames = cfg.rotation.frames
        camera = self.get_camera().to(device)
        statics = self.get_static_cameras(radius=camera.co.radius)
        statics = [camera.to(device) for camera in statics]
        frame_add = 0
        axis = CameraCoordinates._up.to(device)
        if mode in ("horizontal", "both"):
            for frame in self.orbit_frames(camera, axis, rot_frames, statics):
                yield frame
            frame_add += rot_frames
        if mode in ("vertical", "both"):
            axis = axis[[2, 0, 1]]
            for frame in self.orbit_frames(camera, axis, rot_frames, statics):
                yield frame
        return

    def orbit_frames(
        self,
        camera: Camera,
        axis: Float[Tensor, "3"],
        frames: int,
        statics: list[Camera] | None = None,
    ) -> Iterator[Splimage]:
        Q = Quaternion.from_axis_angle(
            axis=axis, angle=torch.tensor(2 * torch.pi / frames, device=axis.device)
        ).to(camera.device)
        for _ in range(frames):
            deformed = self.model.deformed(camera)
            orbit = self.render(camera, deformed=deformed)
            if statics:
                yield self.combine_views(
                    orbit, *(self.render(s, deformed=deformed) for s in statics)
                )
            else:
                yield orbit
            camera.co.Q *= Q

    def get_renderer(self) -> Renderer:
        cfg = self.config
        return Renderer.basic(
            bg_color=torch.tensor(cfg.window.bg_color, device=self.device),
            device=self.device,
        )

    def render(
        self, camera: Camera, deformed: Model | None = None
    ) -> Splimage:
        """Renders `camera` on the deformed model.

        Args:
            camera: Camera to render with; also defines the light (unless the
                lighting alignment is a fixed vector or 'up').
            deformed: Optionally, an already-deformed model to render. When
                omitted, the deformation is evaluated for `camera`. Useful for
                rendering a second view with a deformation evaluated for
                another camera.

        Returns:
            The rendered image as a :class:`Splimage`.
        """
        if deformed is None:
            deformed = self.model.deformed(camera)
        light = self.get_light(camera)
        rgb = self.renderer.render(deformed, camera, light)["render"][:, :3]
        return Splimage(rgb)

    def combine_views(self, *views: Splimage) -> Splimage:
        """Concatenates multiple rendered views side by side.

        Args:
            views: Rendered views in left-to-right order; the first one defines
                the output height.

        Returns:
            A single :class:`Splimage` with all views side by side. Views with
            a smaller height are rescaled to the first view's height while
            preserving their aspect ratio.
        """
        tensors = []
        for view in views:
            t = view.image()
            if t.device != views[0].device:
                t = t.to(views[0].device)
            if t.dtype != views[0].dtype:
                t = t.to(views[0].dtype)
            if t.shape[:2] != views[0].image().shape[:2]:
                raise ValueError(
                    f"Channel or batch mismatch between views: "
                    f"{t.shape} vs {views[0].image().shape}."
                )
            if t.shape[2] != views[0].image().shape[2]:
                # Rescale smaller views to the first view's height, keeping
                # the aspect ratio (pixel F differs between cameras).
                H = views[0].image().shape[2]
                W = round(t.shape[3] * H / t.shape[2])
                t = ImgUtils.resize(t, H=H, W=W)
            tensors.append(t)
        return Splimage(torch.cat(tensors, dim=-1))

    def get_camera(self) -> Camera:
        cfg = self.config.camera
        co = CameraCoordinates(
            Q=Quaternion.from_axis_angle(
                CameraCoordinates._up, torch.tensor(torch.pi / 2)
            ),
            radius=cfg.radius,
        )
        return Camera(H=cfg.H, W=cfg.W, F=cfg.F, co=co)

    def get_static_cameras(self, radius: float | None = None) -> list[Camera]:
        """Builds every static second view from the ``static_view`` list.

        Each entry holds an optional nested ``camera`` config (H, W, F, radius)
        and a `view` preset looking at the world origin. An entry's radius
        defaults to the orbit camera's `radius` argument.

        Args:
            radius: Orbit camera radius, used when an entry's
                ``camera.radius`` is None.

        Returns:
            Static :class:`Camera` list, ordered left-to-right in the output.
            Empty when ``static_view`` is None or an empty list.
        """
        cfg = self.config.static_view
        if cfg is None:
            return []
        if not isinstance(cfg, (list, tuple)):
            raise ValueError(
                "static_view must be a list of view configs "
                "({'view': <preset>, 'camera': {H, W, F, radius}})."
            )
        cameras = []
        for entry in cfg:
            entry = Dict(entry) if isinstance(entry, dict) else None
            if entry is None or "view" not in entry:
                raise ValueError(
                    f"Invalid static view entry '{entry}'; expected a dict with "
                    "a 'view' key and an optional 'camera' config."
                )
            if entry.view not in self._views:
                raise ValueError(
                    f"Invalid static view: '{entry.view}'; expected one of "
                    f"{sorted(self._views)} or None."
                )
            axis, angle = self._views[entry.view]
            cam_cfg = Config(
                {"H": 512, "W": 512, "F": 60, "radius": None, **entry.get("camera", {})},
            )
            entry_radius = cam_cfg.radius if cam_cfg.radius is not None else radius
            if entry_radius is None:
                raise ValueError(
                    f"Static view '{entry.view}' radius is unset; pass the orbit "
                    "camera's radius or set static_view camera radius."
                )
            co = CameraCoordinates(
                Q=Quaternion.from_axis_angle(axis, torch.tensor(angle)),
                radius=entry_radius,
            )
            cameras.append(
                Camera(H=cam_cfg.H, W=cam_cfg.W, F=cam_cfg.F, co=co)
            )
        return cameras

    def get_light(self, camera: Camera) -> LightSource:
        cfg = self.config.lighting
        device = camera.device
        align = cfg.alignment
        strength = torch.tensor(cfg.strength)
        if align == "camera":
            return LightSource.from_camera(camera, strength).to(device)
        elif align == "up":
            return LightSource(strength=strength).to(device)
        elif isinstance(align, (list, tuple)) and len(align) == 3:
            return LightSource(
                torch.tensor(align, device=device, dtype=torch.float32), strength
            ).to(device)
        raise ValueError(
            f"Alignment '{align}' is invalid for lighting alignment config; "
            "expected 'camera', 'up', or a vector of length 3."
        )
