from __future__ import annotations
import torch

from typing import Any, Dict, Tuple, Optional
from torch import Tensor
from jaxtyping import Float, Int

from .base import Model
from .splat import Splat
from ..utils.quaternion import Quaternion
from ..utils.camera import Camera


class Mesh(Model):
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Int[Tensor, "F 3"],
        normalize: bool = True,
    ):
        super().__init__()
        self.F = F
        self.V = V

    def _tensors(self) -> Dict[str, Tensor]:
        return {"V": self.V, "F": self.F}

    def _apply_tensors(self, tensor_dict: Dict[str, Tensor]):
        self.V = tensor_dict["V"]
        self.F = tensor_dict["F"]

    def to_dict(self) -> Dict[str, Any]:
        return {"V": self.V, "F": self.F}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Mesh:
        return cls(V=data["V"], F=data["F"])

    def copy(self) -> Mesh:
        return Mesh(
            V=self.V.clone(),
            F=self.F.clone(),
        )

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
    def centroid(self) -> Float[Tensor, "3"]:
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
    def unique_edges(self) -> Tuple[Int[Tensor, "E 2"], Int[Tensor, "2E"]]:
        return torch.unique(self.sorted_halfedges, dim=0, return_inverse=True)

    @property
    def E(self) -> Int[Tensor, "E 2"]:
        return self.unique_edges[0]

    @property
    def EF(self) -> Tuple[Int[Tensor, "E"], Int[Tensor, "F"]]:
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
    def face_areas(self) -> Float[Tensor, "F"]:
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
        ortho = torch.cross(ab, ac, dim=1)
        with torch.no_grad():
            flip_mask = (ortho * (self.face_centroids - self.centroid)).sum(dim=1) < 0
        return ortho * torch.where(flip_mask[:, None], -1.0, 1.0)

    @property
    def neighbour_count(self) -> Float[Tensor, "V"]:
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
        diag = diag.scatter_add(0, I, W)

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
    def adjacency(self) -> Tuple[Int[Tensor, "V+1"], Int[Tensor, "2E"]]:
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

    def vertex_neighbours(self, v_idx: Int[Tensor, "V"]) -> Int[Tensor, "E"]:
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
        F: Optional[Int[Tensor, "F 3"]] = None,
        face_idx: Optional[Int[Tensor, "N"]] = None,
    ) -> Float[Tensor, "N C"]:
        F = self.F if F is None else F
        if face_idx is not None:
            F = F[face_idx]
        emb_data = data[F]
        b_co = b_co.unsqueeze(-1)
        return (emb_data * b_co).sum(dim=1)

    @property
    def make_wireframe(self) -> Splat:
        E = self.E
        V = self.V
        nV = self.num_V
        nE = self.num_E
        vertex_normals = self.vertex_normals_normalized
        device = self.device
        splat_scale = 0.005
        endpoints = (V + 0.01 * vertex_normals)[E]
        midpoints = endpoints.mean(dim=1)
        means = torch.cat([V, midpoints], dim=0)
        directions = endpoints[:, 0] - endpoints[:, 1]
        lengths = (directions).norm(dim=1)
        scales_endpoints = torch.full_like(V, splat_scale / 10)
        scales_midpoints = (
            torch.tensor([splat_scale / 100] * 3, device=device)
            .unsqueeze(0)
            .repeat(midpoints.shape[0], 1)
        )
        scales_midpoints[:, 2] = lengths * splat_scale * 80
        scales = torch.cat([scales_endpoints, scales_midpoints], dim=0)
        quats_endpoints = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            .unsqueeze(0)
            .expand(nV, -1)
        )
        quats_midpoints = Quaternion.vector_alignment(scales_midpoints, directions)
        quats = torch.cat([quats_endpoints, quats_midpoints], dim=0)
        opacities = torch.tensor([1.0], device=device).expand(nV + nE)
        colors = torch.ones_like(scales).unsqueeze(1)
        return Splat(means, quats, scales, opacities, colors, 0)

    def rasterize(self, camera: Camera) -> Float[Tensor, "B 4 H W"]:
        return self.make_wireframe.rasterize(camera)
