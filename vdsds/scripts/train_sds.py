import logging
import math
import random
from pathlib import Path
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from ..rendering import Camera, CameraCoordinates, LightSource, Renderer
from ..rendering.shader import (
    Albedo,
    Alpha,
    Antialias,
    BackgroundColor,
    Clamp,
    LambdaShader,
    Normal,
    SoftLambertShader,
)
from ..utils.config import Config
from ..utils.deepfloyd import DeepFloydGuidance
from ..utils.img import ImgUtils, Splimage
from ..utils.quaternion import Quaternion
from ..utils.resize_right import resize
from ..utils.spherical_basis import SphericalGaussianBasis
from .analyze import AnalyzeDeformation
from .base import ViewableScript
from .orbit import OrbitFrames

logger = logging.getLogger(__name__)


class TrainModelSDS(ViewableScript):
    _default_config_overrides: ClassVar[dict[str, any]] = {
        "window": {
            "fps": 12,
            "close_on_finish": True,
        },
        "path": {
            "run_dir": None,
        },
        "optim": {
            "epochs": 400,
            "lr": 0.005,
            "accum_steps": 2,
            "seed": None,
            "train_model": False,
            "train_centroids": False,
            "train_sigmas": False,
            "train_weights": True,
        },
        "diffusion": {
            "loss": "bsd",  # Options: 'sds','bsd'
            "model_size": "L",  # Options: 'S', 'M', 'L', 'XL'
            "dtype": "float16",
            "cpu_offload": False,
            "guidance_scale": 100.0,
        },
        "prompt": {
            "text": ["Crocodile", "Moose"],
            "anchors": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            # "text": ["Turtle"],
            # "anchors": [[1.0, 0.0, 0.0]],
            "overlap": 0.5,
            "basis_proc": "norm",  # Options: 'norm', 'square', 'max'
        },
        "loss": {
            "diffusion": 1.0,
            "jacobian": 500.0,
            "laplacian": 0.0,
        },
        "model": {"name": "mesh"},
        "deformation": {
            "name": "vd_full",  # Options: 'full', 'vd_full', 'vd_restrained'
            "num_funcs": 12,
            "init": "fibonacci",  # Options: 'fibonacci', 'random'
            "overlap": 1.0,
            "normalize": False,
        },
        "camera": {
            "H": 512,
            "W": 512,
            "F_min": 40,
            "F_max": 75,
            "radius_min": 1.25,
            "radius_max": 2.5,
            "views": 8,
            "view_source": "orbit",  # Options: 'random', 'orbit'abs
            "elevation_min": 0.0,  # Only used for 'orbit' mode
            "elevation_max": math.pi / 3,  # Only used for 'orbit' mode
            "point_upwards": True,
            "aug_views": 0,
            "aug_type": "jitter",  # Options: 'jitter', 'random'
            "jitter_range": math.pi / 4,
        },
        "shader": {
            "resize": True,
            "down_H": 64,  # Only used if 'resize' is True
            "down_W": 64,  # Only used if 'resize' is True
            "shading_beta": 5.0,
            "bg_color": [0.2, 0.7, 0.0],  # Options: 'random' or provide color.
            "crop_fit": True,
            "crop_border": 16,  # Only used if 'crop_fit' is True
        },
        "lighting": {
            "alignment": "camera",  # Options: 'camera', 'up'
            "jitter_sigma": math.pi / 4,
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
        self.model.train(
            model=config.optim.train_model,
            centroids=config.optim.train_centroids,
            sigmas=config.optim.train_sigmas,
            weights=config.optim.train_weights,
        )
        self.renderer = self.get_renderer()
        accum_steps = config.optim.accum_steps
        prompts, text_embeds, prompt_basis = self.get_text_embeds(df)
        optimizer = torch.optim.Adam(
            [*self.model.get_parameters()],
            lr=config.optim.lr,
        )
        for e in range(config.optim.epochs):
            if self.abort_requested():
                break
            epoch_loss = 0.0
            optimizer.zero_grad()
            cameras = self.camera_batch()
            for _ in range(config.optim.accum_steps):
                loss = (
                    self.diffusion_loss(
                        cameras, df, text_embeds, prompt_basis, len(prompts)
                    )
                    * config.loss.diffusion
                ) / accum_steps
                loss.backward()
                epoch_loss += loss.item()
            j_loss = self.jacobian_loss() * config.loss.jacobian * accum_steps
            l_loss = self.laplacian_loss() * config.loss.laplacian * accum_steps
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

    def get_text_embeds(
        self, df: DeepFloydGuidance
    ) -> tuple[list[str], Float[Tensor, "..."], SphericalGaussianBasis | None]:
        cfg = self.config
        txt = cfg.prompt.text
        views = cfg.camera.views
        aug_views = cfg.camera.aug_views
        batch_size = views * (1 + aug_views)
        if isinstance(txt, list):
            prompts = [p + ", a 3d rendering" for p in txt]
        else:
            prompts = [txt + ", a 3d rendering"]
        if cfg.diffusion.loss == "bsd":
            basis = SphericalGaussianBasis(
                num_funcs=len(txt),
                num_dims=1,
                batch_size=1,
                sigma_overlap=cfg.prompt.overlap,
                centroids=torch.tensor(cfg.prompt.anchors, dtype=torch.float32),
                normalize=True,
            ).to(self.device)
            prompts.append("")
        else:
            basis = None
        text_embeds = df.encode_text_2(
            prompts,
            # negative_prompt=[""] * batch_size,
            batch_size=batch_size,
        )
        return prompts, text_embeds, basis

    def get_prompt_basis(
        self, basis_func: SphericalGaussianBasis, deltas: Float[Tensor, "N 3"]
    ) -> Float[Tensor, "M N"]:
        cfg = self.config.prompt
        proc = cfg.basis_proc
        basis = basis_func._basis(deltas)
        if proc == "norm":
            return basis / basis.norm(dim=-1, keepdim=True)
        elif proc == "square":
            basis = basis**2
            return basis / basis.norm(dim=-1, keepdim=True)
        elif proc == "max":
            ret = torch.zeros_like(basis)
            ret[basis.argmax(dim=-1, keepdim=True)] = 1.0
            return ret
        else:
            raise ValueError(f"Basis processign option '{proc}' is invalid.")

    def diffusion_loss(
        self,
        cameras: list[tuple[Camera, Camera]],
        df: DeepFloydGuidance,
        text_embeds,
        alpha_basis: SphericalGaussianBasis | None,
        num_prompts: int,
    ) -> Float[Tensor, ""]:
        cfg = self.config.diffusion
        renders = self.get_training_renders(cameras)
        if cfg.loss == "sds":
            rt = df.SDS(renders, text_embeds, controller=None)
        else:
            with torch.no_grad():
                deltas = self.model.model.centroid - torch.stack(
                    [dc.location for rc, dc in cameras], dim=0
                )
                attn_alphas = self.get_prompt_basis(alpha_basis, deltas)
            assert renders.shape[0] * (num_prompts + 1) == text_embeds.shape[0], (
                f"render batch {renders.shape[0]} x {num_prompts + 1} branches "
                f"!= embed rows {text_embeds.shape[0]}"
            )
            rt = df.ActvnReplace(
                torch.cat([renders] * (num_prompts + 1), dim=0),
                text_embeds,
                prompt_num=num_prompts,
                controller=None,
                attn_ctrl_alphas=attn_alphas.tolist(),
            )
        return rt["loss_sds"]

    def get_training_renders(
        self, cameras: list[tuple[Camera, Camera]]
    ) -> Float[Tensor, "B 3 H W"]:
        cfg = self.config
        renders = []
        for i, (render_cam, deform_cam) in enumerate(cameras):
            light = self.randomized_light(render_cam)
            render = self.renderer(self.model.deformed(deform_cam), render_cam, light)[
                "render"
            ]
            renders.append(render)
        return torch.cat(renders, dim=0)

    def random_cam(self) -> Camera:
        cfg = self.config.camera
        radius = random.random() * (cfg.radius_max - cfg.radius_min) + cfg.radius_min
        F = random.random() * (cfg.F_max - cfg.F_min) + cfg.F_min
        if cfg.view_source == "random":
            return Camera.random_rot(
                H=cfg.H,
                W=cfg.W,
                F=F,
                radius=radius,
                point_upwards=cfg.point_upwards,
            ).to(self.device)
        elif cfg.view_source == "orbit":
            angle = torch.rand((), device=self.device) * 2 * torch.pi
            Q_orbit = Quaternion.from_axis_angle(
                CameraCoordinates._up.to(self.device), angle
            ).to(self.device)
            cam = Camera(
                H=cfg.H, W=cfg.W, F=F, co=CameraCoordinates(radius=radius, Q=Q_orbit)
            ).to(self.device)
            e_angle = (
                torch.rand((), device=self.device)
                * (cfg.elevation_max - cfg.elevation_min)
                + cfg.elevation_min
            )
            cam.rotate_from_image_space(dx=0, dy=1.0, deg=e_angle)
            return cam

    def jitter_cam(self, camera: Camera) -> Camera:
        cfg = self.config.camera
        cam = camera.copy()
        device = cam.device
        angle_range = torch.pi if cfg.aug_type == "random" else cfg.jitter_range
        angle = (torch.rand((), device=device) * 2 - 1) * angle_range
        if cfg.view_source == "random":
            axis = torch.rand((3,), device=device) - 0.5
            axis /= axis.norm().clamp(min=1e-8)
            Q_rot = Quaternion.from_axis_angle(axis, angle).to(device)
            cam.co.Q *= Q_rot
            if cfg.point_upwards:
                cam = cam.point_upwards()
        elif cfg.view_source == "orbit":
            axis = cam.co._up.to(cam.device)
            Q_rot = Quaternion.from_axis_angle(axis, angle).to(device)
            cam.co.Q *= Q_rot
        return cam

    def camera_batch(self) -> list[tuple[Camera, Camera]]:
        cfg = self.config.camera
        base_cams = [self.random_cam() for _ in range(cfg.views)]
        cameras = []
        for cam in base_cams:
            cameras.append((cam, cam))
            for _ in range(cfg.aug_views):
                view_cam = self.jitter_cam(cam)
                cameras.append((view_cam, cam))
        return cameras

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
                    self.get_training_renders(self.camera_batch())
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

    def laplacian_loss(self) -> torch.Tensor:
        J = torch.einsum("bfij,cfjk->bfik", self.model.jacobians_3d(), self.model.J_src)
        V_disps = self.model.poisson.solve_poisson(J) - self.model.model.V[None]
        B, N, C = V_disps.shape
        Y = (
            torch.sparse.mm(
                self.model.model.L_cotan_csr, V_disps.permute(1, 0, 2).reshape(N, B * C)
            )
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
            axis = torch.rand((3,), device=device) - 0.5
            angle = torch.randn((), device=device) * jitter
            Q_rot = Quaternion.from_axis_angle(axis, angle)
            origin = Q_rot.rotate_vector(origin)
        return LightSource(origin=origin)

    def get_renderer(self) -> Renderer:
        cfg = self.config.shader
        shaders = [
            Albedo(),
            Normal(),
            SoftLambertShader(apply_to={"render"}, beta=cfg.shading_beta),
            Alpha(apply_to={"render"}),
        ]
        if cfg.crop_fit:
            shaders.append(
                LambdaShader(
                    lambda ctx, i: ImgUtils.crop_alpha(
                        i.permute(0, 3, 1, 2),
                        border=16,
                        keep_aspect=True,
                        preserve_size=True,
                    ).permute(0, 2, 3, 1),
                    apply_to={"render"},
                )
            )
        shaders += [
            BackgroundColor(
                color=torch.tensor(cfg.bg_color, device=self.device),
                apply_to={"render"},
            ),
            Clamp(apply_to={"render"}),
            Antialias(apply_to={"render"}),
        ]
        if cfg.resize:
            shaders.append(
                LambdaShader(
                    lambda ctx, i: resize(
                        i.permute(0, 3, 1, 2),
                        out_shape=(cfg.down_H, cfg.down_W),
                    ).permute(0, 2, 3, 1),
                    apply_to={"render"},
                )
            )
        return Renderer(shaders, self.device)
