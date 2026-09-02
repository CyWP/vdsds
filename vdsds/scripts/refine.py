import random

import torch

from typing import List, Optional, Dict, Any
from jaxtyping import Float
from torch import Tensor

from .base import ViewableScript
from ..utils.camera import Camera
from ..utils.quaternion import Quaternion
from ..utils.optim import get_cosine_schedule_with_warmup
from ..deformations.splat_mesh_deform import SplatMeshDeformation
from ..deformations.textured_mesh_deform import TexturedMeshDeformation
from ..representations.textured_mesh import TexturedMesh
from ..representations.splat_mesh import SplatMesh
from ..dc.dc import DC, DCConfig


class RefineModel(ViewableScript):
    """
    DreamCocoon-style refinement: the SDEdit-edited render of each view serves as
    the reference for SDS/DDS training, pushing the learnable deformation toward
    the target prompt.
    """

    _config_defaults = {
        "epochs": 1000,
        "lr": 0.002,
        "warmup": 10,
        "degree": 0,
        "views": 4,
        "edit_rate": 10,
        "edit_count": 1,
        "skip_min_ratio": 0.8,
        "skip_max_ratio": 0.9,
        "log_step": 100,
        "max_grad_norm": 1.0,
        "dc": {},
    }

    def __init__(
        self,
        model_path: str,
        fps: int = 24,
        view: bool = True,
        close_on_finish: bool = False,
        finish_on_close: bool = True,
        device: torch.device = torch.device("cuda:0"),
        config: Dict[str, Any] = {},
        **kwargs,
    ):
        self.config = {**self._config_defaults, **config}
        super().__init__(
            model_path, fps, view, close_on_finish, finish_on_close, device
        )

    def load_model(self, path: str):
        model = super().load_model(path)
        if isinstance(model, TexturedMesh):
            model = TexturedMeshDeformation(model, degree=self.config["degree"])
        if isinstance(model, SplatMesh):
            model = SplatMeshDeformation(model, degree=self.config["degree"])
        return model

    def run(self):
        config = self.config
        cameras = self._get_orbit_cameras(views=config["views"])
        bg_color = torch.tensor([0.5, 0.5, 0.5]).to(self.device).requires_grad_(True)
        dc = DC(
            DCConfig(
                device=self.device,
                sd_pretrained_model_or_path="timbrooks/instruct-pix2pix",
                **config["dc"],
            ),
            use_wandb=False,
        )
        self.model.train(train_model=False)
        self._edits: Dict[int, Float[Tensor, "1 3 H W"]] = {}

        optimizer = torch.optim.Adam(
            [*self.model.parameters(), bg_color], lr=config["lr"]
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, config["warmup"], config["epochs"]
        )

        step = 0
        for e in range(config["epochs"]):
            epoch_loss = 0.0
            optimizer.zero_grad()
            for i, cam in enumerate(cameras):
                # Render the current view.
                tgt_render = self.get_renders([cam], bg=bg_color)

                # Periodically produce a fresh SDEdit-edited reference for this view.
                if step % config["edit_rate"] == 0:
                    edit_img = None
                    for _ in range(config["edit_count"]):
                        edit_img = self.sdedit(dc, tgt_render.detach())
                    self._edits[i] = edit_img.detach()

                ref = self._edits.get(i)
                if ref is None:
                    ref = tgt_render.detach()
                    self._edits[i] = ref

                tgt_x0 = dc.encode_image(tgt_render * 2 - 1)
                ref_x0 = dc.encode_image(ref * 2 - 1)
                ref_dist = dc.encode_src_image(ref)

                loss = dc(tgt_x0, ref_x0, ref_dist)
                loss.backward()
                epoch_loss += loss.item()
                step += 1

            torch.nn.utils.clip_grad_norm_(
                [*self.model.parameters(), bg_color], config["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            print(f"[Epoch {e}] Loss: {epoch_loss}")

    def sdedit(self, dc, img: Float[Tensor, "1 3 H W"]) -> Float[Tensor, "1 3 H W"]:
        """Encode an image to a latent, SDEdit it, and decode back to pixels."""
        input_img = img.to(self.device)
        h, w = input_img.shape[2:]
        l = min(h, w)
        h = int(h * 512 / l)
        w = int(w * 512 / l)
        resized = torch.nn.functional.interpolate(
            input_img, size=(h, w), mode="bilinear"
        )
        latents = dc.encode_image(resized.to(self.device))

        sdedit_steps = 20
        min_step = int(sdedit_steps * self.config["skip_min_ratio"])
        max_step = int(sdedit_steps * self.config["skip_max_ratio"])
        skip = random.randint(min_step, max_step)

        edit_x0 = dc.run_sdedit(
            latents, num_inference_steps=sdedit_steps, skip=skip, image_latent=latents
        )
        edit_img = dc.decode_latent(edit_x0)

        if edit_img.shape[2:] != input_img.shape[2:]:
            edit_img = torch.nn.functional.interpolate(
                edit_img, size=input_img.shape[2:], mode="bilinear"
            )
        return edit_img

    def _get_orbit_cameras(self, views: int = 8) -> List[Camera]:
        cameras = []
        camera = Camera(H=512, W=512).to(self.device)
        cameras.append(camera)
        rot = Quaternion.from_axis_angle(
            torch.tensor([0.0, 1.0, 0.0]), torch.tensor(2 * torch.pi / views)
        ).to(self.device)
        for _ in range(views - 1):
            camera = camera.copy()
            camera.co.Q *= rot
            cameras.append(camera)
        return cameras

    def get_renders(
        self, cameras: List[Camera], bg: Optional[Float[Tensor, "3"]] = None
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
