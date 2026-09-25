from __future__ import annotations

import logging
from typing import Any

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as torch_F
from jaxtyping import Float, Int
from torch import Tensor

from ..utils.camera import Camera
from ..utils.conventions import NVDIFFRAST_CONVERSION_MTX, OPENGL_CONVERSION_MTX, UP
from ..utils.light import LightSource
from .base import Model

logger = logging.getLogger(__name__)


class Mesh(Model):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        texture: Float[Tensor, 3] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.F = F.to(torch.int32).contiguous()
        self.V = V.contiguous()
        d_area = torch.linalg.norm(
            torch.cross(
                self.V[self.F[:, 1]] - self.V[self.F[:, 0]],
                self.V[self.F[:, 2]] - self.V[self.F[:, 0]],
                dim=-1,
            ),
            dim=-1,
        )
        degenerate = d_area <= 0
        if degenerate.any():
            logger.warning(
                f"Mesh has {(degenerate.sum()).item()} degenerate zero-area faces "
                f"(indices {torch.nonzero(degenerate).flatten()[:10].tolist()}); "
                "jacobians and the poisson solve may produce NaNs."
            )
        self.texture = (
            torch.tensor([0.5, 0.5, 0.5], device=V.device).contiguous()
            if texture is None
            else texture.contiguous()
        )
        self.opengl_conversion = torch.tensor(
            OPENGL_CONVERSION_MTX,
            dtype=V.dtype,
            device=V.device,
        )
        self.nvdiffrast_conversion = torch.tensor(
            NVDIFFRAST_CONVERSION_MTX,
            dtype=V.dtype,
            device=V.device,
        )
        self.up = torch.tensor(UP, device=V.device, dtype=torch.float32)
        self.ctx = dr.RasterizeCudaContext()

    def _tensors(self) -> dict[str, Tensor]:
        return {
            "V": self.V,
            "F": self.F,
            "texture": self.texture,
        }

    def _apply_tensors(self, tensor_dict: dict[str, Tensor]):
        self.V = tensor_dict["V"]
        self.F = tensor_dict["F"]
        self.texture = tensor_dict["texture"]

    def to(self, device: torch.device | str) -> Mesh:
        super().to(device)
        self.opengl_conversion = self.opengl_conversion.to(device)
        self.nvdiffrast_conversion = self.nvdiffrast_conversion.to(device)
        self.up = self.up.to(device)
        device = self.device
        if device.type == "cuda":
            self.ctx = dr.RasterizeCudaContext()
        return self

    def _dict_data(self) -> dict[str, Any]:
        return {
            **super()._dict_data(),
            "V": self.V,
            "F": self.F,
            "texture": self.texture,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Mesh:
        return cls(V=data["V"], F=data["F"], texture=data["texture"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mesh:
        return Model.from_dict(data)

    def __len__(self) -> int:
        return self.num_V

    @property
    def num_V(self) -> int:
        return self.V.shape[0]

    @property
    def num_E(self) -> int:
        return self.E.shape[0]

    @property
    def num_F(self) -> int:
        return self.F.shape[0]

    @property
    def centroid(self) -> Float[Tensor, 3]:
        return torch.mean(self.V, dim=0)

    @property
    def halfedges(self) -> Int[Tensor, "2E 2"]:
        F = self.F
        e01 = F[:, [0, 1]]
        e12 = F[:, [1, 2]]
        e20 = F[:, [2, 0]]

        return torch.cat([e01, e12, e20], dim=0)

    @property
    def sorted_halfedges(self) -> Int[Tensor, "2E 2"]:
        return torch.sort(self.halfedges, dim=1).values

    @property
    def unique_edges(self) -> tuple[Int[Tensor, "E 2"], Int[Tensor, "2E"]]:
        return torch.unique(self.sorted_halfedges, dim=0, return_inverse=True)

    @property
    def E(self) -> Int[Tensor, "E 2"]:
        return self.unique_edges[0]

    @property
    def EF(self) -> tuple[Int[Tensor, E], Int[Tensor, F]]:
        nF = self.num_F
        device = self.device
        E, inv = self.unique_edges
        face_idx = torch.arange(nF, device=device).repeat(3)
        perm = torch.argsort(inv)
        inv_sorted = inv[perm]
        face_sorted = face_idx[perm]
        counts = torch.bincount(inv_sorted, minlength=E.shape[0])
        indptr = torch.zeros(E.shape[0] + 1, device=device, dtype=torch.long)
        indptr[1:] = torch.cumsum(counts, dim=0)
        return indptr, face_sorted

    @property
    def volume(self) -> Float[Tensor, ""]:
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        return torch.sum(
            torch.cross(b - a, c - a, dim=1) * (a - self.centroid).norm() / 6
        )

    @property
    def face_centroids(self) -> Float[Tensor, "F 3"]:
        return self.V[self.F].mean(dim=1)

    @property
    def face_areas(self) -> Float[Tensor, F]:
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        ab = b - a
        ac = c - a
        ortho = torch.cross(ab, ac, dim=1)
        return torch.norm(ortho, dim=1) * 0.5

    @property
    def face_normals(self) -> Float[Tensor, "F 3"]:
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        ab = b - a
        ac = c - a
        return torch.cross(ab, ac, dim=1)

    @property
    def neighbour_count(self) -> Float[Tensor, V]:
        counts = torch.zeros(self.V.shape[0], device=self.V.device)
        counts = counts.index_add(
            0, self.F.view(-1), torch.ones(self.F.numel(), device=self.V.device)
        )
        return counts

    @property
    def vertex_normals(self) -> Float[Tensor, "V 3"]:
        normals = torch.zeros_like(self.V)
        face_normals = self.face_normals
        normals.index_add_(0, self.F.view(-1), face_normals.repeat(1, 3).view(-1, 3))
        counts = self.neighbour_count.clamp_min(1.0).unsqueeze(1)
        normals = normals / counts
        return normals

    @property
    def vertex_normals_normalized(self) -> Float[Tensor, "V 3"]:
        normals = self.vertex_normals
        return normals / normals.norm(dim=1, keepdim=True)

    @property
    def grad_operator(self) -> Float["F V 3"]:
        """
        V: (V, 3) vertex positions
        F: (F, 3) face indices

        Returns:
            G: sparse COO tensor of shape (F, V, 3)
            such that G @ vertex_scalars -> per-face gradients.
        """
        V = self.V
        F = self.F
        p0 = V[F[:, 0]]
        p1 = V[F[:, 1]]
        p2 = V[F[:, 2]]

        e1 = p1 - p0
        e2 = p2 - p0

        n = torch.cross(e1, e2, dim=-1)
        area2 = n.norm(dim=-1, keepdim=True)

        grad_phi0 = torch.cross(n, p2 - p1, dim=-1) / area2
        grad_phi1 = torch.cross(n, p0 - p2, dim=-1) / area2
        grad_phi2 = torch.cross(n, p1 - p0, dim=-1) / area2

        # (F, 3, 3)
        values = torch.stack(
            [
                grad_phi0,
                grad_phi1,
                grad_phi2,
            ],
            dim=1,
        )

        # Indices for (face, vertex, xyz)
        face_idx = torch.arange(F.shape[0], device=F.device)[:, None, None].expand(
            -1, 3, 3
        )

        vertex_idx = F[:, :, None].expand(-1, -1, 3)

        xyz_idx = torch.arange(3, device=F.device)[None, None, :].expand(
            F.shape[0], 3, -1
        )

        indices = torch.stack(
            [
                face_idx.reshape(-1),
                vertex_idx.reshape(-1),
                xyz_idx.reshape(-1),
            ]
        )

        G = torch.sparse_coo_tensor(
            indices,
            values.reshape(-1),
            size=(F.shape[0], V.shape[0], 3),
            device=V.device,
            dtype=V.dtype,
        )

        return G.coalesce()

    @property
    def L_cotan(self) -> Float[Tensor, "V V"]:
        F = self.F
        V = self.V
        eps = 1e-8
        n = V.shape[0]
        device = V.device

        i, j, k = F[:, 0], F[:, 1], F[:, 2]
        vi, vj, vk = V[i], V[j], V[k]

        v_ji = vj - vi
        v_ki = vk - vi
        v_ij = vi - vj
        v_kj = vk - vj
        v_ik = vi - vk
        v_jk = vj - vk

        cot_i = (v_ji * v_ki).sum(axis=1) / (
            torch.norm(torch.cross(v_ji, v_ki), dim=1) + eps
        )
        cot_j = (v_ij * v_kj).sum(axis=1) / (
            torch.norm(torch.cross(v_ij, v_kj), dim=1) + eps
        )
        cot_k = (v_ik * v_jk).sum(axis=1) / (
            torch.norm(torch.cross(v_ik, v_jk), dim=1) + eps
        )

        W = torch.cat([cot_i, cot_i, cot_j, cot_j, cot_k, cot_k]) * 0.5
        I = torch.cat([j, k, k, i, i, j])
        J = torch.cat([k, j, i, k, j, i])

        L = torch.sparse_coo_tensor(
            torch.stack([I, J]), W, (n, n), device=device
        ).coalesce()

        diag = torch.zeros(n, device=device)
        diag = diag.scatter_add(0, I.to(torch.int64), W)

        M = torch.sparse_coo_tensor(
            torch.stack([torch.arange(n), torch.arange(n)]),
            diag,
            (n, n),
            device=device,
            is_coalesced=True,
        )

        return (L - M).coalesce()

    @property
    def L_cotan_csr(self) -> Float[Tensor, "V V"]:
        L = self.L_cotan.to_sparse_csr()
        return L

    @property
    def L_cotan_dense(self) -> Float[Tensor, "V V"]:
        L = self.L_cotan.to_dense()
        return L

    @property
    def adjacency(self) -> tuple[Int[Tensor, V + 1], Int[Tensor, "2E"]]:
        F = self.F
        nV = self.num_V
        i = torch.cat([F[:, 0], F[:, 1], F[:, 2]])
        j = torch.cat([F[:, 1], F[:, 2], F[:, 0]])
        i = torch.cat([i, j])
        j = torch.cat([j, i])

        perm = torch.argsort(i)
        i_sorted = i[perm]
        j_sorted = j[perm]

        counts = torch.bincount(i_sorted, minlength=nV)
        indptr = torch.zeros(nV + 1, dtype=torch.long, device=F.device)
        indptr[1:] = torch.cumsum(counts, dim=0)

        return indptr, j_sorted

    @property
    def adjacency_coo(self) -> Float[Tensor, "V V"]:
        E = self.E
        num_V = self.num_V
        idx = torch.stack(
            (torch.cat((E[:, 0], E[:, 1])), torch.cat((E[:, 1], E[:, 2])))
        )
        vals = torch.ones((E.shape[0],), device=self.device)
        A = torch.sparse_coo_tensor(
            idx, vals, (num_V, num_V), device=self.device, is_coalesced=True
        )
        return A

    @property
    def adjacency_csr(self) -> Float[Tensor, "V V"]:
        A = self.adjacency
        return A.to_sparse_csr()

    @property
    def adjacency_dense(self) -> Float[Tensor, "V V"]:
        A = self.adjacency
        return A.to_dense()

    @property
    def L_umbrella(self) -> Float[Tensor, "V V"]:
        F = self.F
        device = self.device
        num_V = F.max()
        diag_idx = torch.arange(0, num_V, 1, device=device, dtype=torch.long)
        A = self.adjacency
        A_idx = A.indices()
        N = self.neighbour_count.unsqueeze(-1).expand(1, num_V)
        vals = A[A_idx] / N[A_idx]
        idx = torch.cat((A_idx, torch.stack(diag_idx, diag_idx)), dim=0)
        vals = torch.cat((vals, torch.full(diag_idx.size, -1.0, device=device)))
        L = torch.sparse_coo_tensor(idx, vals, (num_V, num_V), is_coalesced=True)
        return L

    @property
    def L_umbrella_csr(self) -> Float[Tensor, "V V"]:
        L = self.L_umbrella.to_sparse_csr()
        return L

    @property
    def L_umbrella_dense(self) -> Float[Tensor, "V V"]:
        L = self.L_umbrella.to_sparse_dense()
        return L

    @property
    def euler_number(self) -> int:
        return self.num_V + self.num_F - self.num_E

    @property
    def genus(self) -> int:
        return self.euler_number // 2

    @property
    def manifold(self) -> bool:
        if self.euler_number % 2 != 0:
            return False
        F = self.F
        e01 = F[:, [0, 1]]
        e12 = F[:, [1, 2]]
        e20 = F[:, [2, 0]]

        edges = torch.cat([e01, e12, e20], dim=0)
        return torch.unique(torch.sort(edges, dim=1).values, dim=0)

    @torch.no_grad
    def collapse_edge(self, idx: int) -> Mesh:
        V = self.V
        E = self.E
        F = self.F
        num_V = self.num_V
        e = E[idx]
        i1, i2 = e
        self.V[i1] = self.V[e].mean(dim=1)
        V_mask = torch.ones(num_V, dtype=torch.bool)
        V_mask[i2] = 0
        self.V = V[V_mask]
        has_i1 = (F == i1).any(dim=1)
        has_i2 = (F == i2).any(dim=1)
        has_both = has_i1 & has_i2
        has_2 = has_2 != has_both
        F[has_2 != has_both, 0] = i1
        self.F = F[has_both]
        return self

    def vertex_normal_alignment(self, vec: Float[Tensor, "N 3"]) -> Float[Tensor, N]:
        return (self.vertex_normals_normalized * vec).sum(dim=1)

    def vertex_neighbours(self, v_idx: Int[Tensor, V]) -> Int[Tensor, E]:
        A_ptr, A_vals = self.adjacency
        return A_vals[A_ptr[v_idx] : A_ptr[v_idx + 1]]

    def collapse_is_manifold(self, edge_idx: int) -> bool:
        E = self.E
        nbhds = self.vertex_neighbours(E[edge_idx])
        nbhd_set = nbhds.flatten().unique()
        return nbhds.numel() - nbhd_set.numel() == 2

    @torch.no_grad
    def collapse_edges(self, idx: torch.Tensor) -> Mesh:
        for i in range(len(idx)):
            ei = idx[i]
            if self.collapse_manifold(ei):
                self.collapse_edge(ei)
                idx[idx > ei] -= 1
        return self

    @torch.no_grad
    def split_faces(self, idx: torch.Tensor) -> Mesh:
        F = self.F
        V = self.V
        nV = self.num_V
        device = self.device
        nNew = idx.shape[0]
        sel_F = F[idx]
        new_midpoints = V[sel_F].mean(dim=1)
        new_V = torch.cat([V, new_midpoints], dim=0)
        new_V_idx = torch.arange(nV, nV + nNew, device=device, dtype=torch.long)
        base = sel_F.repeat(3, 1)
        cols = torch.arange(3, device=device).repeat_interleave(nNew)
        rows = torch.arange(3 * nNew, device=device)
        vals = new_V_idx.repeat(3)
        new_sel_F = base.index_put((rows, cols), vals)
        new_F = torch.cat([F, new_sel_F[nNew:]])
        new_F[idx] = new_sel_F[:nNew]
        self.V = new_V
        self.F = new_F
        return self

    def barycentric_interpolate(
        self,
        data: Float[Tensor, "V C"],
        b_co: Float[Tensor, "N 3"],
        F: Int[Tensor, "F 3"] | None = None,
        face_idx: Int[Tensor, N] | None = None,
    ) -> Float[Tensor, "N C"]:
        F = self.F if F is None else F
        if face_idx is not None:
            F = F[face_idx]
        emb_data = data[F]
        b_co = b_co.unsqueeze(-1)
        return (emb_data * b_co).sum(dim=1)

    def camera_to_nvdiffrast(self, camera):
        device = self.device
        dtype = camera.w2c.dtype

        # Rotate the camera's orbital coordinate system:
        #
        # camera's default position:
        #     (0, 0, -r)
        #
        # becomes:
        #     (0, -r, 0)
        #
        # This is a +90° rotation around world X.

        # Your camera convention:
        #   +X right
        #   +Y down
        #   +Z forward
        #
        # OpenGL:
        #   +X right
        #   +Y up
        #   -Z forward
        gl_conversion = torch.diag(
            torch.tensor(
                [1.0, -1.0, -1.0, 1.0],
                dtype=dtype,
                device=device,
            )
        )

        fx = camera.Fx
        fy = camera.Fx

        projection = torch.zeros(
            (4, 4),
            dtype=dtype,
            device=device,
        )

        projection[0, 0] = 2 * fx / camera.W
        projection[1, 1] = -2 * fy / camera.H

        zn = camera.Zn
        zf = camera.Zf

        projection[2, 2] = -(zf + zn) / (zf - zn)
        projection[2, 3] = -2 * zf * zn / (zf - zn)
        projection[3, 2] = -1

        return projection @ gl_conversion @ camera.w2c @ self.nvdiffrast_conversion

    @property
    def VH(self) -> Float[Tensor, "V 4"]:
        return torch.cat(
            [self.V, torch.ones(self.V.shape[0], 1, device=self.V.device)], dim=1
        )

    def raster_albedo(self, rast) -> Float[Tensor, "B H W 4"]:
        return dr.interpolate(
            self.texture[None].expand(self.num_V, 3).contiguous(), rast, self.F
        )[0]

    def lambert_shade(
        self,
        rast,
        albedo_map: Float[Tensor, "B H W 4"],
        camera: Camera,
        light: LightSource,
    ) -> Float[Tensor, "B H W 4"]:
        normal_map, _ = dr.interpolate(self.vertex_normals_normalized, rast, self.F)
        pos_map, _ = dr.interpolate(self.V, rast, self.F)
        ray_map = light.origin[None, None, None] - pos_map
        return (
            albedo_map
            * light.strength
            * (ray_map * normal_map).sum(dim=-1, keepdim=True)
        )

    def half_lambert_shade(
        self,
        rast,
        albedo_map: Float[Tensor, "B H W 4"],
        camera: Camera,
        light: LightSource,
    ) -> Float[Tensor, "B H W 4"]:
        normal_map, _ = dr.interpolate(self.vertex_normals_normalized, rast, self.F)
        pos_map, _ = dr.interpolate(self.V, rast, self.F)
        ray_map = light.origin[None, None, None] - pos_map
        return (
            albedo_map
            * light.strength
            * ((ray_map * normal_map).sum(dim=-1, keepdim=True) / 2 + 0.5)
        )

    def to_nvdiffrast(self, vals: Float[Tensor, "N C"]) -> Float[Tensor, "... C"]:
        N, C = vals.shape
        return vals @ self.nvdiffrast_conversion[:C, :C]

    def soft_lambert_shade(
        self,
        rast,
        albedo_map: Float[Tensor, "B H W 4"],
        camera: Camera,
        light: LightSource,
        beta: float = 5.0,
    ) -> Float[Tensor, "B H W 4"]:
        normal_map, _ = dr.interpolate(self.vertex_normals_normalized, rast, self.F)
        pos_map, _ = dr.interpolate(self.V, rast, self.F)
        ray_map = light.origin[None, None, None] - pos_map
        align_map = (ray_map * normal_map).sum(dim=-1, keepdim=True)
        return albedo_map * light.strength * (torch_F.softplus(align_map, beta=beta))

    def rasterize(
        self,
        camera: Camera,
        light: LightSource,
        shading: str = "soft",
        antialias: bool = True,
    ) -> Float[Tensor, "B 4 H W"]:
        pos_clip = self.VH @ self.camera_to_nvdiffrast(camera).T
        rast, _ = dr.rasterize(self.ctx, pos_clip[None], self.F, [camera.H, camera.W])
        albedo = self.raster_albedo(rast)
        if shading == "lambert":
            rgb = self.lambert_shade(rast, albedo, camera, light)
        elif shading == "soft":
            rgb = self.soft_lambert_shade(rast, albedo, camera, light)
        elif shading == "half":
            rgb = self.half_lambert_shade(rast, albedo, camera, light)
        rgba = torch.cat(
            [rgb, torch.ones((*rgb.shape[:3], 1), device=rgb.device)], dim=-1
        )
        rgba = torch.where(rast[..., 3:4] > 0, rgba, torch.zeros_like(rgba))
        if antialias:
            pos_clip_batched = pos_clip[None] if pos_clip.ndim == 2 else pos_clip
            rgba = dr.antialias(rgba, rast, pos_clip_batched, self.F)
        return rgba.permute(0, 3, 1, 2).clamp(0, 1)

    @classmethod
    def from_mesh_data(
        cls,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        unit_box: bool = True,
        **kwargs,
    ) -> Mesh:
        if unit_box:
            V_min, V_max = V.min(dim=0).values, V.max(dim=0).values
            V_extent = V_max - V_min
            V_center = (V_max + V_min) / 2
            V = (V - V_center) / V_extent.max()
        V = torch.stack([V[:, 1], V[:, 0], -V[:, 2]], dim=1)
        return cls(V, F)
