import logging
import math
from pathlib import Path
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.camera import Camera
from ..utils.config import Config
from ..utils.deepfloyd import DeepFloydGuidance
from ..utils.img import Splimage
from ..utils.light import LightSource
from ..utils.quaternion import Quaternion
from .analyze import AnalyzeDeformation
from .base import ViewableScript
from .orbit import OrbitFrames

logger = logging.getLogger(__name__)


class TrainModelSDS(ViewableScript):
    _default_config_overrides: ClassVar[dict[str, any]] = {
        "window": {"fps": 12},
        "path": {
            "run_dir": None,
        },
        "optim": {
            "epochs": 400,
            "lr": 0.005,
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
        "loss": {
            "sds": 1.0,
            "jacobian": 500.0,
            "laplacian": 10000.0,
        },
        "model": {"name": "mesh"},
        "deformation": {
            "name": "vd_restrained",  # Options: 'full', 'vd_full', 'vd_restrained'
            "num_funcs": 6,
            "init": "fibonacci",  # Options: 'fibonacci', 'random'
            "overlap": 2.0,
            "normalize": False,
        },
        "camera": {
            "views": 16,
            "point_upwards": True,
            "radius": 1.5,
            "bg_color": [0.1, 0.7, 0.0],  # Options: 'random' or provide color.
        },
        "lighting": {
            "alignment": "camera",  # Options: 'camera', 'up'
            "jitter_sigma": math.pi / 10,
        },
        "post": {
            "orbit": True,
            "check_batch_views": 2,
            "analyze": True,
        },
    }

    def run(self):
        device = self.device
        config = self.config
        df = DeepFloydGuidance(config.diffusion, device)
        torch.manual_seed(config.optim.seed)
        self.model.train(train_model=False)
        ref_L = self.model.model.L_cotan
        txt = config.diffusion.prompt
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
        n_views = config.camera.views
        accum_steps = config.optim.accum_steps
        n_samples = n_views * accum_steps
        optimizer = torch.optim.Adam(
            [*self.model.parameters()],
            lr=config.optim.lr,
        )
        for e in range(config.optim.epochs):
            if self.abort_requested():
                break
            epoch_loss = 0.0
            optimizer.zero_grad()
            cameras = self.camera_batch()
            for _ in range(config.optim.accum_steps):
                for i, cam in enumerate(cameras):
                    tgt_render = self.get_renders([cam], bg=self.get_bg_color())
                    # Splimage(tgt_render.clone().detach()).save(path / f"{i:03}.png")
                    loss = (
                        df.SDS(tgt_render, text_embeds, controller=None)["loss_sds"]
                        / n_samples
                        * config.loss.sds
                    )
                    loss.backward()
                    epoch_loss += loss.item()
            j_loss = self.jacobian_loss() * config.loss.jacobian * accum_steps
            l_loss = self.laplacian_loss(ref_L) * config.loss.laplacian * accum_steps
            (j_loss + l_loss).backward()
            optimizer.step()
            self.log(
                e,
                {
                    "reconstruction_loss": epoch_loss,
                    "jacobian_loss": j_loss.item(),
                    "laplacian_loss": l_loss.item(),
                },
            )

    def camera_batch(self) -> list[Camera]:
        config = self.config
        return [
            Camera.random_rot(
                H=224,
                W=224,
                radius=config.camera.radius,
                point_upwards=config.camera.point_upwards,
            ).to(self.device)
            for _ in range(config.camera.views)
        ]

    def log(self, epoch: int, data: dict[str, any]):
        if not hasattr(self, "_logs"):
            self._logs = {}
        self._logs[str(epoch)] = data
        printlog = f"[Epoch {epoch}]:\n"
        for k, v in data.items():
            printlog += f"\t{k}: {v}\n"
        print(printlog)

    def finish(self):
        cfg = self.config
        run_dir = Path(cfg.path.run_dir)
        run_dir.mkdir(exist_ok=True, parents=True)
        deformation_file = run_dir / "deformation.vd3d"
        logs_file = run_dir / "logs.yaml"
        config_file = run_dir / "config.yaml"
        cfg.save(config_file)
        Config(self._logs).save(logs_file)
        self.model.save(deformation_file)
        bv = cfg.post.check_batch_views
        if bv > 0:
            print("Saving sample batch renders...")
            batch_dir = run_dir / "batch_test"
            batch_dir.mkdir(exist_ok=True, parents=True)
            for b in range(bv):
                renders = Splimage(
                    self.get_renders(self.camera_batch(), bg=self.get_bg_color())
                ).to_pil()
                if not isinstance(renders, list):
                    renders = [renders]
                for i, r in enumerate(renders):
                    Splimage(r).save(batch_dir / f"batch_{(i + b * bv):05}.png")

        if cfg.post.orbit:
            print("Generating orbit video...")
            orbit_cfg = {
                "model": cfg.model,
                "path": {
                    "model": deformation_file,
                    "out_dir": run_dir,
                    "file_name": "orbit",
                },
            }
            orbit_script = OrbitFrames(self.device, config=Config(orbit_cfg))
            orbit_script.run()
            orbit_script.finish()
        if cfg.post.analyze:
            print("Analyzing deformation...")
            analyze_cfg = {
                "model": cfg.model,
                "path": {
                    "model": deformation_file,
                    "out_dir": run_dir,
                },
            }
            analyze_script = AnalyzeDeformation(self.device, config=Config(analyze_cfg))
            analyze_script.run()
            analyze_script.finish()

    def jacobian_loss(self) -> torch.Tensor:
        J_def = self.model.J_deform
        if hasattr(J_def, "weights"):
            return (self.model.J_deform.weights**2).mean()
        return (self.model.J_deform**2).mean()

    def laplacian_loss(self, ref_L) -> torch.Tensor:
        J = torch.einsum("bfij,cfjk->bfik", self.model.jacobians_3d(), self.model.J_src)
        V_disps = self.model.poisson.solve_poisson(J) - self.model.model.V[None]
        B, N, C = V_disps.shape
        Y = (
            torch.sparse.mm(ref_L, V_disps.permute(1, 0, 2).reshape(N, B * C))
            .reshape(N, B, C)
            .permute(1, 0, 2)
        )
        return (Y**2).mean()

    def randomized_light(self, camera: Camera) -> LightSource:
        align = self.config.lighting.alignment
        jitter = self.config.lighting.jitter_sigma
        device = camera.device
        if align == "camera":
            origin = LightSource.from_camera(camera).origin
        elif align == "up":
            origin = torch.tensor(LightSource._up, device=device, dtype=torch.float32)
        else:
            raise ValueError(
                f"Alignment '{align}' is invalid for lighting alignment config."
            )
        if jitter != 0.0:
            axis = torch.rand((3,), device=device)
            axis /= axis.norm()
            angle = torch.randn((), device=device) * jitter
            Q_rot = Quaternion.from_axis_angle(axis, angle)
            origin = Q_rot.rotate_vector(origin)
        return LightSource(origin=origin)

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
        bg = self.config.camera.bg_color
        if isinstance(bg, list):
            return torch.tensor(bg, device=self.device, dtype=self.dtype)
        elif bg == "random":
            return torch.rand(3, device=self.device, dtype=self.dtype)
        else:
            raise ValueError(f"Invalid background color value: '{bg}'.")
