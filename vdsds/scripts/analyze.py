from pathlib import Path
from typing import ClassVar

import torch

from ..deformations.base import Deformation
from ..representations.vertextured_mesh import VerTexturedMesh
from ..utils.config import Config
from .base import ViewableScript
from .orbit import OrbitFrames


class AnalyzeDeformation(ViewableScript):
    """
    Loads a model (or deformation) without opening the viewer, and renders
    frames of a camera orbiting the model. The camera rotates using a
    :class:`Quaternion` at every frame, and each frame is written as an image
    in the output directory. The orbit can be horizontal, vertical, or both.
    """

    _default_config_overrides: ClassVar[Config] = Config(
        {
            "window": {
                "view": False,
            },
            "path": {
                "model": None,
                "out_dir": "./",
                "suffix": "analyzed",
            },
            "model": {
                "color_0": [1.0, 0.0, 0.0],
                "color_1": [0.0, 1.0, 0.0],
            },
            "post": {"orbit": True},
        }
    )

    def run(self):
        cfg = self.config
        out_dir = Path(cfg.path.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        model_path = Path(cfg.path.model)
        suffix = cfg.path.suffix
        out_path = out_dir / f"{model_path.stem}_{suffix}.vd3d"
        new_model = self.model.copy()
        tex_mesh, stats = self.textured_mesh_from_basis_weights()
        new_model.model = tex_mesh
        new_model.save(out_path)
        stats.save(out_dir / f"{model_path.stem}_{suffix}.yaml")
        if cfg.post.orbit:
            orbit_cfg = {
                "model": {**cfg.model, "name": "vertextured_mesh"},
                "path": {
                    "model": out_path,
                    "out_dir": out_dir,
                    "file_name": f"orbit_{suffix}",
                },
                "lighting": {
                    "alignment": "camera",
                    "strength": 1.0,
                },
            }
            orbit_script = OrbitFrames(self.device, config=Config(orbit_cfg))
            orbit_script.run()
            orbit_script.finish()

    @torch.no_grad()
    def textured_mesh_from_basis_weights(self) -> tuple[VerTexturedMesh, Config]:
        assert isinstance(self.model, Deformation), (
            "Analyzed model must have a deformation."
        )
        deformation = self.model.J_deform
        assert not deformation.batched_basis, (
            "Cannot analyze deformation with a batched basis."
        )
        device = self.model.device
        cfg = self.config
        color_0, color_1 = (
            torch.tensor(cfg.model.color_0, device=device),
            torch.tensor(cfg.model.color_1, device=device),
        )
        mesh = self.model.model
        u = mesh.centroid - mesh.V
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        basis = deformation._basis(u).sum(dim=-1)
        bmin, bmax = basis.min(), basis.max()
        bn = (basis - bmin) / (bmax - bmin).clamp(min=1e-8)
        tex = bn[:, None] * color_1[None] + (1 - bn)[:, None] * color_0[None]
        analyzed_mesh = VerTexturedMesh(mesh.V, mesh.F, tex)
        stats = Config(
            {
                "basis": {
                    "min": bmin.item(),
                    "max": bmax.item(),
                    "mean": basis.mean().item(),
                }
            }
        )
        return analyzed_mesh, stats
