import torch

from typing import List, Optional, Dict, Any
from jaxtyping import Float
from torch import Tensor

from .base import ViewableScript
from ..utils.camera import Camera
from ..utils.quaternion import Quaternion
from ..utils.optim import get_cosine_schedule_with_warmup
from ..deformations.splat_mesh_deform import SplatMeshDeformation
from ..dc.dc import DC, DCConfig
from ..utils.img import Splimage


class TrainModel(ViewableScript):
    _config_defaults = {
        "epochs": 1000,
        "lr": 0.01,
        "warmup": 10,
        "degree": 0,
        "views": 4,
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
        if not isinstance(model, SplatMeshDeformation):
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
        with torch.no_grad():
            src_renders = self.get_renders(cameras)
            ref_L = self.model.model.mesh.L_cotan_csr
        optimizer = torch.optim.Adam(
            [*self.model.parameters(), bg_color], lr=config["lr"]
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, config["warmup"], config["epochs"]
        )
        for e in range(config["epochs"]):
            epoch_loss = 0.0
            optimizer.zero_grad()
            with torch.no_grad():
                sr = self.apply_bg(src_renders, bg_color)
                src_x0 = dc.encode_image(sr)
                src_dist = [dc.encode_src_image(s.unsqueeze(0)) for s in sr]
            for i, cam in enumerate(cameras):
                tgt_render = self.get_renders([cam], bg=bg_color)
                tgt_x0 = dc.encode_image(tgt_render)
                loss = dc(tgt_x0, src_x0[i].unsqueeze(0), src_dist[i])
                loss.backward()
                epoch_loss += loss.item()
            loss = 1.0 * self.cotan_loss(ref_L)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            scheduler.step()
            print(f"[Epoch {e}] Loss: {epoch_loss}")

    def cotan_loss(self, ref_L) -> torch.Tensor:
        w = self.model.V_deform.weights
        loss = torch.tensor(0.0, device=w.device)
        for i in range(w.shape[1]):
            loss += (((ref_L @ w[:, i]) ** 2) * 2**i).mean()
        return loss

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
