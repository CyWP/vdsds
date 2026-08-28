from __future__ import annotations
import torch

from typing import Any, Dict, Optional
from torch import Tensor
from jaxtyping import Float, Int, Bool

from .base import Model
from .mesh import Mesh
from .splat import Splat
from ..utils.quaternion import Quaternion
from ..utils.camera import Camera
from ..utils.img import Splimage


class SplatMesh(Model):
    def __init__(
        self,
        mesh: Mesh,
        faces: Int[Tensor, "N"],
        barys: Float[Tensor, "N 3"],
        disps: Float[Tensor, "N"],
        scales: Float[Tensor, "N 2"],
        rots: Float[Tensor, "N"],
        colors: Float[Tensor, "N C"],
        sh_degree: int,
        show_mesh: bool = True,
        splat_height: float = 0.0001,
    ):
        super().__init__()
        self.mesh = mesh
        self.faces = faces
        self.barys = barys
        self.disps = disps
        self.scales = scales
        self.rots = rots
        self.colors = colors
        self.sh_degree = sh_degree
        self.up = torch.tensor([0.0, 0.0, -1.0])
        self.show_mesh = show_mesh
        self.splat_height = torch.tensor(splat_height)

    def _tensors(self) -> Dict[str, Tensor]:
        return {
            "faces": self.faces,
            "barys": self.barys,
            "disps": self.disps,
            "scales": self.scales,
            "rots": self.rots,
            "colors": self.colors,
            "up": self.up,
            "splat_height": self.splat_height,
        }

    def _apply_tensors(self, tensor_dict: Dict[str, Tensor]):
        self.faces = tensor_dict["faces"]
        self.barys = tensor_dict["barys"]
        self.disps = tensor_dict["disps"]
        self.scales = tensor_dict["scales"]
        self.rots = tensor_dict["rots"]
        self.colors = tensor_dict["colors"]
        self.up = tensor_dict["up"]
        self.splat_height = tensor_dict["splat_height"]

    def to(self, *args, **kwargs) -> SplatMesh:
        super().to(*args, **kwargs)
        self.mesh = self.mesh.to(*args, **kwargs)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mesh": self.mesh.to_dict(),
            "faces": self.faces,
            "barys": self.barys,
            "disps": self.disps,
            "scales": self.scales,
            "rots": self.rots,
            "colors": self.colors,
            "sh_degree": self.sh_degree,
            "show_mesh": self.show_mesh,
            "splat_height": self.splat_height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SplatMesh:
        mesh = Mesh.from_dict(data["mesh"])
        return cls(
            mesh=mesh,
            faces=data["faces"],
            barys=data["barys"],
            disps=data["disps"],
            scales=data["scales"],
            rots=data["rots"],
            colors=data["colors"],
            sh_degree=int(data["sh_degree"]),
            show_mesh=data.get("show_mesh", True),
            splat_height=float(data.get("splat_height", 0.0001)),
        )

    def __len__(self) -> int:
        return self.scales.shape[0]

    def copy(self) -> SplatMesh:
        return SplatMesh(
            self.mesh.copy(),
            self.faces.clone(),
            self.barys.clone(),
            self.disps.clone(),
            self.scales.clone(),
            self.rots.clone(),
            self.colors.clone(),
            self.sh_degree,
            self.show_mesh,
            float(self.splat_height),
        )

    @property
    def opacities(self) -> Float[Tensor, "N"]:
        return torch.ones((1,), device=self.device, dtype=self.dtype).expand(
            (len(self),)
        )

    @property
    def centroid(self) -> Float[Tensor, "3"]:
        return self.mesh.centroid

    @property
    def scales_3d(self) -> Float[Tensor, "N 3"]:
        return torch.cat(
            [
                self.scales,
                self.splat_height.unsqueeze(-1)
                .to(self.device)
                .unsqueeze(-1)
                .expand(self.faces.shape[0], 1),
            ],
            dim=1,
        )

    @property
    def splat_areas(self) -> Float[Tensor, "N"]:
        return self.scales[:, 0] * self.scales[:, 1]

    @property
    def reference_axis(self) -> Float[Tensor, "N 3"]:
        return self.up.unsqueeze(0).expand(len(self), -1).to(self.device)

    @property
    def anchors_3d(self) -> Float[Tensor, "N 3"]:
        return self.mesh.barycentric_interpolate(
            self.mesh.V, self.barys, face_idx=self.faces
        )

    @property
    def normals_3d(self) -> Float[Tensor, "N 3"]:
        return self.mesh.barycentric_interpolate(
            self.mesh.vertex_normals, self.barys, face_idx=self.faces
        )

    @property
    def disps_3d(self) -> Float[Tensor, "N 3"]:
        return self.normals_3d * self.disps.unsqueeze(-1)

    @property
    def means_3d(self) -> Float[Tensor, "N 3"]:
        return self.anchors_3d + self.disps_3d

    @property
    def quats_3d(self) -> Float[Tensor, "N 4"]:
        return Quaternion.tensor_rotation(
            Quaternion.vector_alignment(self.reference_axis, self.normals_3d),
            Quaternion.axis_rotations(self.normals_3d, self.rots),
        )

    @property
    def make_splat(self) -> Splat:
        return Splat(
            self.means_3d,
            self.quats_3d,
            self.scales_3d,
            self.opacities,
            self.colors.unsqueeze(1),
            self.sh_degree,
        )

    def rasterize(self, camera: Camera) -> Float[Tensor, "1 4 H W"]:
        return self.make_splat.rasterize(camera)

    @classmethod
    def from_mesh_data(
        cls,
        V: Float[Tensor, "V"],
        F: Float[Tensor, "F"],
        uv_co: Float[Tensor, "3F 2"],
        uv_idx: Float[Tensor, "3F"],
        texture: Splimage,
        num_splats: int = 100000,
        unit_box: bool = True,
    ):
        if unit_box:
            V_min, V_max = V.min(dim=0).values, V.max(dim=0).values
            V_extent = V_max - V_min
            V_center = (V_max + V_min) / 2
            V = (V - V_center) / V_extent.max()
        V = torch.stack([V[:, 1], V[:, 2], V[:, 0]], dim=1)
        mesh = Mesh(V, F)
        areas = mesh.face_areas
        total_area = areas.sum()
        splats_per_face = torch.round(areas / total_area * num_splats).to(torch.long)
        n_splat = splats_per_face.sum().item()
        faces = torch.repeat_interleave(torch.arange(0, F.shape[0], 1), splats_per_face)
        r = torch.rand((n_splat,))
        r_sq = torch.sqrt(torch.rand((n_splat,)))
        barys = torch.stack([1 - r_sq, r_sq * (1 - r), r_sq * r], dim=1)
        disps = torch.zeros((n_splat,))
        scales = torch.full((n_splat, 2), (total_area / n_splat).item() ** 0.5)
        rots = torch.zeros((n_splat,))
        uv_co = torch.stack([uv_co[:, 1], uv_co[:, 0]], dim=1)
        emb = uv_co[uv_idx[faces]]
        colors = (
            texture.image_sample((emb * barys.unsqueeze(-1)).sum(dim=1))[0, :, :3] * 2.5
            - 1.25
        )
        return SplatMesh(
            mesh, faces, barys, disps, scales, rots, colors, sh_degree=0
        )
