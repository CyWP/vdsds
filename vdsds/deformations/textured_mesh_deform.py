from __future__ import annotations

import torch
import torch.nn as nn

from typing import Dict
from torch import Tensor

from ..utils.harmonics import SphericalHarmonic
from ..utils.camera import Camera
from ..utils.img import ImgUtils
from .base import Deformation
from ..representations.textured_mesh import TexturedMesh


class TexturedMeshDeformation(Deformation):
    def __init__(self, model: TexturedMesh, degree: int = 2):
        super().__init__(model)
        self.V_deform = SphericalHarmonic(
            degree=degree, num_dims=3, batch_size=model.num_V
        )
        B, C, H, W = model.texture.shape
        self.register_parameter(
            "tex_deform", nn.Parameter(torch.zeros((B, C, H // 8, W // 8)))
        )
        self.register_buffer("degree", torch.tensor(degree))

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> TexturedMeshDeformation:
        model_keys = {}
        v_deform_keys = {}
        colors_deform_keys = {}
        direct_keys = {}

        for key, value in state_dict.items():
            if key.startswith("model."):
                model_keys[key[6:]] = value
            elif key.startswith("V_deform."):
                v_deform_keys[key[9:]] = value
            else:
                direct_keys[key] = value

        model = TexturedMesh.from_state_dict(model_keys)
        degree = int(direct_keys.get("degree", 2))

        instance = cls(model=model, degree=degree)
        instance.V_deform = SphericalHarmonic.from_state_dict(v_deform_keys)
        instance.colors_deform = SphericalHarmonic.from_state_dict(colors_deform_keys)
        return instance

    def deformed(self, camera: Camera) -> TexturedMesh:
        camera_loc = camera.location.unsqueeze(0)
        m = self.model
        delta = m.centroid - camera_loc
        V_disp = self.V_deform.from_cartesian(delta)
        B, C, H, W = m.texture.shape
        return TexturedMesh(
            V=m.V + V_disp,
            F=m.F,
            uv_co=m.uv_co,
            texture=m.texture + ImgUtils.resize(self.tex_deform, H, W),
        )
