from __future__ import annotations
import torch

from typing import Tuple, Optional
from torch import nn, Tensor
from jaxtyping import Float, Int, Bool

from .base import Model
from .mesh import Mesh
from .splat import Splat
from ..utils.quaternion import Quaternion
from ..utils.camera import Camera


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
        splat_height: float = 0.001,
    ):
        super().__init__()
        self.mesh = mesh
        self.register_parameter("faces", faces)
        self.register_parameter("barys", barys)
        self.register_parameter("disps", disps)
        self.register_parameter("scales", scales)
        self.register_parameter("rots", rots)
        self.register_parameter("colors", colors)
        self.register_buffer("sh_degree", sh_degree)
        self.register_buffer("up", torch.tensor([0.0, 0.0, -1.0]))
        self.show_mesh = show_mesh
        self.register_buffer("splat_height", torch.tensor(splat_height))

    def __len__(self) -> int:
        return self.scales.shape[0]

    def copy(self) -> SplatMesh:
        return SplatMesh(
            self.mesh,
            self.faces,
            self.barys,
            self.disps,
            self.scales,
            self.rots,
            self.colors,
            self.sh_degree,
            self.show_mesh,
        )

    @property
    def opacity(self) -> Float[Tensor, "N"]:
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
                .unsqueeze(-1)
                .expand(self.faces.shape[0], 1),
            ],
            dim=1,
        )

    @property
    def splat_areas(self) -> Float[Tensor, "N"]:
        """This metric is dependent on area, but does not precisely represent area"""
        return self.scales[:, 0] * self.scales[:, 1]

    @property
    def reference_axis(self) -> Float[Tensor, "N 3"]:
        return self.up.unsqueeze(0).expand(len(self), -1)

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
        """
        Computes means, scales, and quats.
        """
        return Splat(
            self.means_3d,
            self.quats_3d,
            self.scales_3d,
            self.opacity,
            self.colors.unsqueeze(1),
            self.sh_degree,
        )

    def rasterize(self, camera: Camera) -> Float[Tensor, "1 4 H W"]:
        return self.make_splat.rasterize(camera)

    def depth_map(
        self, camera: Camera, alpha: bool = True
    ) -> (
        Float[Tensor, "1 1 H W"]
        | Tuple[Float[Tensor, "1 1 H W"], Float[Tensor, "1 1 H W"]]
    ):
        return self.make_splat.depth_map(camera, alpha=alpha)

    def render(self, camera: Camera) -> ImageTensor:
        splat = self.make_splat
        self.show_mesh = False
        if self.show_mesh:
            mesh = self.mesh.make_wireframe
            distances = (camera.location().unsqueeze(0) - mesh.means).norm(dim=1)
            d_min, d_max, d_mean = (
                distances.min(),
                distances.max(),
                distances.mean(),
            )
            weights = ((1 - ((distances - d_min) / (d_max - d_min))) * 2 - 1).clamp(
                0, 1
            )
            mesh.opacities = mesh.opacities * weights
            splat = Splat.combine([splat, mesh])
        render = self.make_splat.render(camera)
        if self.show_mesh:
            mesh = self.mesh.make_wireframe
            distances = (camera.location().unsqueeze(0) - mesh.means).norm(dim=1)
            d_min, d_max, d_mean = (
                distances.min(),
                distances.max(),
                distances.mean(),
            )
            weights = ((1 - ((distances - d_min) / (d_max - d_min))) * 2 - 1).clamp(
                0, 1
            )
            mesh.opacities = mesh.opacities * weights
            mesh_render = mesh.render(camera)
            alpha_mask = mesh_render[:, 3].unsqueeze(1)
            render = alpha_mask + render * (1 - alpha_mask)
        return render

    def render_depth(self, camera: Camera) -> ImageTensor:
        return self.make_splat.render_depth(camera)

    def split_splats(
        self,
        idx: Int[Tensor, "M"],
        scale_factor: float = 0.625,
        split_dirs: Optional[Float[Tensor, "M 3"]] = None,
    ) -> SplatMesh:
        if split_dirs is None:
            split_dirs = (
                torch.randn((idx.shape[0], 3), device=self.device)
                * self.splat_areas.unsqueeze(-1)
                * 0.5
            )
        new_barys = torch.cat([self.barys, self.barys[idx] + split_dirs], dim=0)
        new_barys[idx] -= split_dirs
        self.barys = new_barys
        new_scales = torch.cat([self.scales, self.scales[idx] * scale_factor], dim=0)
        new_scales[idx] *= scale_factor
        self.scales = new_scales
        # self.faces = torch.cat([self.faces, self.>

    @torch.no_grad
    def clone_splats(
        self,
        idx: Int[Tensor, "M"],
        split_dirs: Optional[Float[Tensor, "M 3"]] = None,
    ) -> SplatMesh:
        return self.split_splats(idx, scale_factor=1.0, split_dirs=split_dirs)

    @torch.no_grad
    def filter_splats(self, mask: Bool[Tensor, "N"]) -> SplatMesh:
        assert mask.shape[0] == len(self) and len(mask.shape) == 1
        self.faces = self.faces[mask]
        self.barys = self.barys[mask]
        self.scales = self.scales[mask]
        self.disps = self.disps[mask]
        self.rots = self.rots[mask]
        self.colors = self.colors[mask]
        return self

    @torch.no_grad
    def delete_splats(self, idx: Int[Tensor, "M"]) -> SplatMesh:
        mask = torch.ones((len(self),), device=self.device, dtype=torch.bool)
        mask[idx] = False
        return self.filter_splats(mask)

    @torch.no_grad
    def split_faces(self, idx: Int[Tensor, "M"]) -> SplatMesh:
        mesh = self.mesh.split_faces(idx)
        faces = self.faces
        for i in idx:
            mask = faces == i
            a, b, c = mesh.F[i]
            # Todo: just find trio that has no negative coordinates

    def from_mesh(mesh: Mesh, num_splats: int = 300000) -> SplatMesh:
        device = mesh.device
        F = mesh.F
        areas = mesh.face_areas
        total_area = mesh.face_areas.sum()
        splats_per_face = torch.round(areas / total_area * num_splats).to(torch.long)
        num_splats = splats_per_face.sum()
        faces = torch.repeat_interleave(
            torch.arange(0, F.shape[0], 1, device=device), splats_per_face
        )
        splat_areas = areas[faces]
        barys = torch.rand((num_splats, 3), device=device)
        barys = barys / barys.sum(dim=1, keepdim=True)
        disps = torch.zeros((num_splats,), device=device)
        scales = torch.full((num_splats, 2), 0.25, device=device) * torch.sqrt(
            splat_areas
        ).unsqueeze(1)
        rots = torch.zeros((num_splats,), device=device)
        if mesh.texture is not None:
            uv = mesh.uv_co
            # uv[:, 1] = 1 - uv[:, 1]
            emb = uv[mesh.uv_idx[faces]]
            uv_co = (emb * barys.unsqueeze(-1)).sum(dim=1)
            colors = mesh.texture_sample(uv_co)[0, :, :3] * 2 - 1
            if colors.shape[1] == 4:
                colors = colors[:, :3]  # * colors[:, 3].unsqueeze(-1)
        else:
            colors = torch.rand_like(barys)
        sh_degree = 0
        return SplatMesh(
            device,
            mesh,
            faces,
            barys,
            disps,
            scales,
            rots,
            colors,
            sh_degree,
            show_mesh=True,
        )
