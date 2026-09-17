import math
from typing import Any, ClassVar

import torch
from easydict import EasyDict as edict
from jaxtyping import Float
from torch import Tensor

from ..deformations.mesh_jacobian_deform import MeshJacobianDeformation
from ..utils.camera import Camera
from ..utils.deepfloyd import DeepFloydGuidance
from ..utils.quaternion import Quaternion
from .base import ViewableScript


class TrainModelSDS(ViewableScript):
    _config_defaults: ClassVar = {
        "epochs": 200,
        "lr": 0.025,
        "sds_alpha": 1.0,
        "jacobian_alpha": 25.0,
        "accum_steps": 2,
        "model_size": "M",
        "dtype": "float16",
        "degree": 1,
        "start_degree": 1,
        "views": 4,
        "max_grad": 0.1,
        "cpu_offload": False,
        "guidance_scale": 7.5,
        "seed": 42,
        "num_jitters": 2,
        "jitter_sigma": math.pi / 60,
        "start_color_fit": 80,
    }

    def __init__(
        self,
        model_path: str,
        fps: int = 10,
        view: bool = True,
        close_on_finish: bool = False,
        finish_on_close: bool = True,
        device: torch.device = torch.device("cuda:0"),
        config: dict[str, Any] = {},
        **kwargs,
    ):
        self.config = edict(**{**self._config_defaults, **config})
        super().__init__(
            model_path, fps, view, close_on_finish, finish_on_close, device
        )

    def load_model(self, path: str):
        return MeshJacobianDeformation(
            super().load_model(path),
            degree=self.config["degree"],
            start_degree=self.config["start_degree"],
        )

    def run(self):
        device = self.device
        config = self.config
        cameras = self._get_orbit_cameras(views=config["views"])
        bg_color = torch.tensor([0.25, 0.25, 0.25]).to(self.device).requires_grad_(True)
        self.set_background(bg_color)
        optim_color = torch.tensor(
            [0.65, 0.65, 0.65], device=self.model.device
        ).requires_grad_(True)
        df = DeepFloydGuidance(config, device)
        accum_steps = config["accum_steps"]
        generator = torch.Generator(device=self.model.device)
        generator.manual_seed(config["seed"])
        self.model.train(train_model=False)
        self.model.color_deform.requires_grad_(False)
        og_color = self.model.model.color
        sds_alpha = config["sds_alpha"]
        jacobian_alpha = config["jacobian_alpha"]
        txt = config.prompt
        n_views = config["views"]
        start_color_fit = config["start_color_fit"]
        if isinstance(txt, list):
            prompts = [
                p + ", a 3d rendering" if i != len(txt) - 1 else p
                for i, p in enumerate(txt)
            ]
        else:
            prompts = [txt + ", a 3d rendering"]
        # with torch.no_grad():
        #     ref_L = self.model.model.L_cotan_csr
        print("Target text prompt:", txt)
        text_embeds = df.encode_text_2(prompts, negative_prompt=[""], batch_size=1).to(
            device
        )
        prompt_num = len(prompts)
        n_samples = n_views * (1 + config["num_jitters"]) * accum_steps
        optimizer = torch.optim.Adam(
            [*self.model.parameters(), bg_color, optim_color], lr=config["lr"]
        )
        for e in range(config["epochs"]):
            if (
                e >= start_color_fit
                and not self.model.color_deform.weights.requires_grad
            ):
                self.model.color_deform.requires_grad_(True)
                self.model.model.color = og_color
            else:
                self.model.model.color = optim_color[None].repeat(
                    self.model.model.num_V, 1
                )
            epoch_loss = 0.0
            optimizer.zero_grad()
            for _ in range(accum_steps):
                for cam in [*cameras, *self._jitter_cameras(cameras)]:
                    tgt_render = self.get_renders([cam], bg=bg_color)
                    loss = (
                        df.SDS(tgt_render, text_embeds, controller=None)["loss_sds"]
                        / n_samples
                        * sds_alpha
                    )
                    loss.backward()
                    epoch_loss += loss.item()
            loss = self.jacobian_loss() * jacobian_alpha
            # torch.nn.utils.clip_grad_value_(
            #     [*self.model.parameters(), bg_color], config["max_grad"]
            # )
            optimizer.step()
            print(
                f"[Epoch {e}] Reconstruction Loss: {epoch_loss}, Jacobian loss: {loss.item()}, Background color: {bg_color.clone().detach().tolist()}"
            )

    def jacobian_loss(self) -> torch.Tensor:
        J_def = self.model.J_deform
        W = J_def.weights
        loss = torch.tensor(0.0, device=W.device)
        start = 0
        for i in range(J_def.degree + 1):
            end = start + 2 * i + 1
            loss += (W[:, :, start : end + 1] ** 2 * 2 ** (i)).mean()
            start = end
        return loss

    def cotan_loss(self, ref_L) -> torch.Tensor:
        w = self.model.V_deform.weights
        loss = torch.tensor(0.0, device=w.device)
        for i in range(w.shape[1]):
            loss += (((ref_L @ w[:, i]) ** 2) * 2**i).mean()
        return loss

    def _get_orbit_cameras(self, views: int = 8) -> list[Camera]:
        cameras = []
        camera = Camera(H=224, W=224).to(self.device)
        camera.co.radius += 0.5
        cameras.append(camera)
        rot = Quaternion.from_axis_angle(
            torch.tensor([0.0, 1.0, 0.0]), torch.tensor(2 * torch.pi / views)
        ).to(self.device)
        for _ in range(views - 1):
            camera = camera.copy()
            camera.co.Q *= rot
            cameras.append(camera)
        return cameras

    def _jitter_cameras(self, cameras: list[Camera]) -> list[Camera]:
        """
        For every camera, produce ``num_jitters`` copies rotated by a random
        small rotation (random axis, Gaussian angle with std ``jitter_sigma``
        in radians). Fresh rotations are sampled on every call.
        """
        jittered = []
        for cam in cameras:
            for _ in range(self.config["num_jitters"]):
                jcam = cam.copy()
                axis = torch.randn(3)
                axis = axis / axis.norm().clamp_min(1e-8)
                angle = torch.randn(()) * self.config["jitter_sigma"]
                jcam.co.Q *= Quaternion.from_axis_angle(axis, angle).to(jcam.device)
                jittered.append(jcam)
        return jittered

    def get_renders(
        self, cameras: list[Camera], bg: Float[Tensor, "3"] | None = None
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
