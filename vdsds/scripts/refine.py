import torch
import math

from typing import List, Optional
from jaxtyping import Float
from torch import Tensor

from .base import ViewableScript
from ..utils.camera import Camera
from ..utils.quaternion import Quaternion
from ..utils.optim import get_cosine_schedule_with_warmup
from ..deformations.splat_mesh_deform import SplatMeshDeformation
from ..dc.dc import DC, DCConfig


class RefineModel(ViewableScript):
    def run(self, config: DCConfig, degree: int = 1):
        if not isinstance(self.model, SplatMeshDeformation):
            self.model = SplatMeshDeformation(self.model, degree=degree)
        cameras = self._get_orbit_cameras()
        bg_color = (
            torch.tensor([0.5, 0.5, 0.5]).to(self.model.device).requires_grad_(True)
        )
        dc = DC(config, use_wandb=False)
        with torch.no_grad():
            src_renders = self.get_renders(cameras)
        optimizer = torch.optim.Adam(self.model.get_parameters(), lr=5e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 50, 1000)
        while True:
            optimizer.zero_grad()
            with torch.no_grad():
                src_x0 = dc.encode_image(self.apply_bg(src_renders, bg_color))
                src_dist = dc.encode_src_image(src_renders)
            tgt_x0 = dc.encode_image(self.get_renders(cameras, bg=bg_color))
            loss = dc(tgt_x0, src_x0, src_dist)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Loss: {loss.item()}")

    def _get_orbit_cameras(self, views: int = 8) -> List[Camera]:
        cameras = []
        camera = Camera(H=512, W=512)
        cameras.append(camera)
        rot = Quaternion.from_axis_angle(
            torch.tensor([0.0, 0.0, 0.1]), 2 * math.pi / views
        ).to(self.model.device)
        for _ in range(views - 1):
            camera = camera.copy()
            camera.co.Q *= rot
            cameras.append(camera)
        return cameras

    def get_renders(
        self, cameras: List[Camera], bg: Optional[Float[Tensor, "3"]]
    ) -> Float[Tensor, "B 3 H W"]:
        renders = torch.cat([self.model.rasterize(cam) for cam in cameras], dim=0)
        if bg is not None:
            return self.apply_bg(renders, bg)
        return renders

    def apply_bg(
        self, renders: Float[Tensor, "B 4 H W"], bg: Float[Tensor, "3"]
    ) -> Float[Tensor, "B 3 H W"]:
        B, C, H, W = renders.shape
        alpha = renders[:, 3].unsqueeze(1)
        renders = alpha * renders[:, :3] + (1 - alpha) * bg[None, :, None, None].expand(
            B, -1, H, W
        )
        return renders
