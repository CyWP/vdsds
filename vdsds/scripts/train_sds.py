import logging
import math
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.camera import Camera
from ..utils.deepfloyd import DeepFloydGuidance
from ..utils.light import LightSource
from ..utils.quaternion import Quaternion
from .base import ViewableScript

logger = logging.getLogger(__name__)


class TrainModelSDS(ViewableScript):
    _config_defaults: ClassVar[dict[str, any]] = {
        "optim": {
            "epochs": 400,
            "lr": 0.075,
            "accum_steps": 2,
            "seed": 42,
        },
        "diffusion": {
            "prompt": "Rhinoceros",
            "model_size": "M",  # Options: 'S', 'M', 'L', 'XL'
            "dtype": "float16",
            "cpu_offload": False,
            "guidance_scale": 7.5,
        },
        "losses": {
            "sds": 1.0,
            "jacobian": 500.0,
            "laplacian": 10000.0,
        },
        "model": {"name": "mesh"},
        "deformation": {
            "name": "vd_restrained",  # Options: 'full', 'vd_full', 'vd_restrained'
            "num_funcs": 8,
            "init": "fibonacci",  # Options: 'fibonacci', 'random'
            "overlap": 2.0,
        },
        "camera": {
            "num_views": 16,
            "align_up": True,
            "bg_color": [0.1, 0.7, 0.0],  # Options: 'random' or provide color.
        },
        "lighting": {
            "alignment": "camera",  # Options: 'camera', 'up'
            "jitter_sigma": math.pi / 10,
        },
    }

    def run(self):
        device = self.device
        config = self.config
        df = DeepFloydGuidance(config, device)
        accum_steps = config["accum_steps"]
        generator = torch.Generator(device=self.model.device)
        generator.manual_seed(config["seed"])
        self.model.train(train_model=False)
        # self.model.J_deform.log_sigmas.requires_grad_(False)
        sds_alpha = config["sds_alpha"]
        jacobian_alpha = config["jacobian_alpha"]
        laplacian_alpha = config["laplacian_alpha"]
        ref_L = self.model.model.L_cotan
        txt = config.prompt
        n_views = config["views"]
        if isinstance(txt, list):
            prompts = [
                p + ", a 3d rendering" if i != len(txt) - 1 else p
                for i, p in enumerate(txt)
            ]
        else:
            prompts = [txt + ", a 3d rendering"]
        print("Target text prompt:", txt)
        text_embeds = df.encode_text_2(prompts, negative_prompt=[""], batch_size=1).to(
            device
        )
        n_samples = n_views * accum_steps
        optimizer = torch.optim.Adam(
            [*self.model.parameters()],
            lr=config["lr"],
        )
        cameras = self._get_orbit_cameras(views=n_views)
        from pathlib import Path

        path = Path("./view_test")
        path.mkdir(exist_ok=True)
        for e in range(config["epochs"]):
            epoch_loss = 0.0
            optimizer.zero_grad()
            cameras = [
                Camera.random_rot(H=224, W=224, radius=1.5, point_upwards=True).to(
                    self.device
                )
                for _ in range(n_views)
            ]
            for _ in range(accum_steps):
                for i, cam in enumerate(cameras):
                    # for cam in [*cameras, *self._jitter_cameras(cameras)]:
                    tgt_render = self.get_renders([cam], bg=bg_color)
                    # Splimage(tgt_render.clone().detach()).save(path / f"{i:03}.png")
                    loss = (
                        df.SDS(tgt_render, text_embeds, controller=None)["loss_sds"]
                        / n_samples
                        * sds_alpha
                    )
                    loss.backward()
                    epoch_loss += loss.item()
            j_loss = self.jacobian_loss() * jacobian_alpha * accum_steps
            l_loss = self.laplacian_loss(ref_L) * laplacian_alpha * accum_steps
            (j_loss + l_loss).backward()
            optimizer.step()
            print(
                f"[Epoch {e}]\nReconstruction Loss: {epoch_loss},\nJacobian loss: {j_loss.item()},\nLaplacian loss: {l_loss.item()},\nBackground color: {bg_color.clone().detach().tolist()}"
            )

    def finish(self):
        pass

    def jacobian_loss(self) -> torch.Tensor:
        J_def = self.model.J_deform
        W = J_def.weights
        return (W**2).mean()

    def laplacian_loss(self, ref_L) -> torch.Tensor:
        W = self.model.J_deform.weights
        B, N, D = W.shape
        W = W.permute(1, 0, 2).reshape(N, -1, 3, 3)
        J = W + torch.eye(3, device=W.device, dtype=W.dtype)[None, None]
        J = torch.einsum("bfij,cfjk->bfik", J, self.model.J_src)
        V_disps = self.model.poisson.solve_poisson(J) - self.model.model.V[None]
        B, N, C = V_disps.shape
        Y = (
            torch.sparse.mm(ref_L, V_disps.permute(1, 0, 2).reshape(N, B * C))
            .reshape(N, B, C)
            .permute(1, 0, 2)
        )
        return (Y**2).mean()

    def randomized_light(
        self, camera: Camera, jitter_sigma: float = 0.5
    ) -> LightSource:
        device = camera.device
        axis = torch.rand((3,), device=device)
        axis /= axis.norm()
        angle = torch.randn((), device=device) * jitter_sigma
        Q_rot = Quaternion.from_axis_angle(axis, angle)
        new_location = Q_rot.rotate_vector(
            camera.location[[0, 2, 1]] * torch.tensor([-1.0, -1.0, 1.0], device=device)
        )
        return LightSource(origin=new_location)

    def get_renders(
        self, cameras: list[Camera], bg: Float[Tensor, "3"] | None = None
    ) -> Float[Tensor, "B 3 H W"]:
        renders = torch.cat(
            [self.model.rasterize(cam, self.randomized_light(cam)) for cam in cameras],
            dim=0,
        )
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

    def get_bg_color(self) -> Float[Tensor, "3"]:
        if hasattr(self, "bg_color"):
            return self.bg_color
        bg = self.config.camera.bg_color
