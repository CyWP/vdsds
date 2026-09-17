from __future__ import annotations

import logging
from typing import Any

import numpy
import scipy.sparse
import torch
import torch.nn.functional as torch_F
from cholespy import CholeskySolverD, MatrixType
from jaxtyping import Float, Integer
from torch import Tensor

_logger = logging.getLogger(__name__)


class PoissonSystem:
    def __init__(
        self,
        V: Float[Tensor, "V 3"],
        F: Integer[Tensor, "M 3"],
        grad: Float[Tensor, "3F V"],
        L: Float[Tensor, "V-1 V-1"],
        rhs: Float[Tensor, "V-1 3F"],
        W: Float[Tensor, "F 3 2"],
    ) -> None:
        self.device = V.device
        self.__V = V
        self.__F = F
        self.grad = grad
        self.L = L
        self.rhs = rhs
        self.W = W
        self.my_splu = None

    @classmethod
    def from_mesh(
        cls,
        V: Float[Tensor, "V 3"],
        F: Integer[Tensor, "M 3"],
    ) -> PoissonSystem:
        """Builds the poisson system for a given mesh.

        Conventions used by a PoissonSystem (see remove_first_coo and
        _predicted_jacobians_to_vertices_via_poisson_solve):

            - vertex 0 is the pinned unknown; its row/column is dropped from
              the Laplacian and its row from the rhs (rhs rows are vertex
              major, its columns are block-major: index c*n_faces + f).
            - the solve rearranges flat jacobian rows (3f + c) into
              component-major blocks before multiplying with the rhs.

        Args:
            V: vertex positions (N, 3).
            F: face indices (M, 3).

        Returns:
            A PoissonSystem ready to compute jacobians and solve.

        Raises:
            AssertionError: if face indices are out of bounds or the
                reduced matrices have unexpected shapes.
        """
        grad = gradient_operator(V, F)
        d_area = _stacked_double_areas(V, F)
        d_area_np = d_area.cpu().numpy()
        degenerate = d_area[::3] <= 0
        if degenerate.any():
            _logger.warning(
                f"Mesh has {(degenerate.sum()).item()} degenerate zero-area faces; "
                "the poisson system may be ill-conditioned."
            )
        coo = grad.coalesce()
        grad = torch.sparse_coo_tensor(
            coo.indices(),
            coo.values() / d_area[coo.indices()[0]],
            coo.shape,
            device=V.device,
            dtype=V.dtype,
        ).coalesce()
        sc_grad = _torch_coo_to_scipy(grad)
        nf = F.shape[0]
        packed_of = numpy.zeros(3 * nf, dtype=numpy.int64)
        for c in range(3):
            for f in range(nf):
                packed_of[3 * f + c] = c * nf + f
        sc_grad = scipy.sparse.coo_matrix(
            (sc_grad.data, (packed_of[sc_grad.row], sc_grad.col)), shape=sc_grad.shape
        )
        mass_diag = scipy.sparse.diags(d_area_np)
        device, dtype = V.device, V.dtype
        laplace = _scipy_coo_to_torch(
            (sc_grad.T @ mass_diag @ sc_grad).tocoo(), device, dtype
        )
        rhs = _scipy_coo_to_torch((sc_grad.T @ mass_diag).tocoo(), device, dtype)
        laplace = remove_first_coo(laplace, row=True, col=True)
        rhs = remove_first_coo(rhs, row=True, col=False)
        basis = tangential_basis(V, F)
        assert F.min() >= 0 and F.max() < V.shape[0]
        assert laplace.shape == (V.shape[0] - 1, V.shape[0] - 1)
        assert rhs.shape == (V.shape[0] - 1, 3 * nf)
        return cls(V, F, grad, laplace, rhs, basis)

    def jacobians_from_vertices(
        self, V: Float[Tensor, "B V 3"]
    ) -> Float[Tensor, "B F 3 3"]:
        """Computes full per-face jacobians from vertex positions.

        Args:
            V: batched vertex positions.

        Returns:
            Per-face jacobians of the transformation from the origin mesh.
        """
        res = _multiply_sparse_2d_by_dense_3d(self.grad, V).type_as(V)
        res = res.unsqueeze(2)
        return res.view(V.shape[0], -1, 3, 3).transpose(2, 3)

    def solve_poisson(
        self, jacobians: Float[Tensor, "B F 3 3"]
    ) -> Float[Tensor, "B V 3"]:
        """Solves the poisson system for a batch of face jacobians.

        Args:
            jacobians: per-face jacobians of the transformation.

        Returns:
            The reconstructed vertex positions, mean-centered per batch item.
        """
        if self.my_splu is None:
            self.my_splu = _coo_to_cholesky(self.L)
        sol = _predicted_jacobians_to_vertices_via_poisson_solve(
            self.my_splu,
            self.rhs,
            jacobians.transpose(2, 3)
            .reshape(jacobians.shape[0], -1, 3, 1)
            .squeeze(3)
            .contiguous(),
        )
        return sol - torch.mean(sol, axis=1).unsqueeze(1)

    def restrict_jacobians(
        self, D: Float[Tensor, "B F 3 3"]
    ) -> Float[Tensor, "B F 3 2"]:
        """Restricts full 3D jacobians to the mesh tangential basis.

        Args:
            D: per-face jacobians.

        Returns:
            Per-face 2D jacobians expressed in the face tangential basis.
        """
        return torch.einsum("abcd,bde->abce", (D, self.W.type_as(D)))

    def restricted_jacobians_from_vertices(
        self, V: Float[Tensor, "B V 3"]
    ) -> Float[Tensor, "B F 3 3"]:
        return self.restrict_jacobians(self.jacobians_from_vertices(V))

    def to(self, device: torch.device, **kwargs) -> PoissonSystem:
        """Moves all tensors to the given device.

        Args:
            device: device to move the tensors to.

        Returns:
            The (same) poisson system, moved to `device`.
        """
        self.__V = self.__V.to(device)
        self.__F = self.__F.to(device)
        self.grad = self.grad.to(device)
        self.L = self.L.to(device)
        self.rhs = self.rhs.to(device)
        self.W = self.W.to(device)
        self.my_splu = None
        self.device = device
        return self


class SPLUSolveLayer(torch.autograd.Function):
    """
    Implements the Poisson solve as a differentiable layer,
    with a forward and backward function.
    """

    @staticmethod
    def forward(
        ctx: Any, solver: CholeskySolverD, b: Float[Tensor, "B V-1 3"]
    ) -> Float[Tensor, "B V-1 3"]:
        """Solves the linear system defined by the solver for a given rhs.

        Args:
            ctx: context object (to keep the solver for the backward pass).
            solver: the cholesky solver of the (reduced) poisson system.
            b: right hand side, could be a vector or matrix.

        Returns:
            The vector or matrix x which holds solver.solve(b) = x.
        """
        assert isinstance(b, torch.Tensor)
        assert b.shape[-1] >= 1 and b.shape[-1] <= 3, (
            f"got shape {b.shape} expected last dim to be in range 1-3"
        )
        b = b.contiguous()
        ctx.solver = solver
        vertices = SPLUSolveLayer.solve(solver, b).type_as(b)
        assert not torch.isnan(vertices).any(), (
            "Nan in the forward pass of the POISSON SOLVE"
        )
        return vertices

    def backward(
        ctx: Any, grad_output: Float[Tensor, "B V-1 3"]
    ) -> tuple[None, Float[Tensor, "B V-1 3"]]:
        """Back-propagates through the (symmetric) solve.

        Args:
            ctx: context object holding the solver from the forward pass.
            grad_output: the gradient to be back-propagated.

        Returns:
            The outgoing gradient, obtained by solving with the transposed
            operator of the linear layer M = A^{-1}.
        """

        assert isinstance(grad_output, torch.Tensor)
        assert grad_output.shape[-1] >= 1 and grad_output.shape[-1] <= 3, (
            f"got shape {grad_output.shape} expected last dim to be in range 1-3"
        )
        # when backpropping, if a layer is linear with matrix M, x ---> Mx, then the backprop of gradient g is M^Tg
        # in our case M = A^{-1}, so the backprop is to solve x = A^-T g.
        # Because A is symmetric we simply solve A^{-1}g without transposing, but this will break if A is not symmetric.
        grad_output = grad_output.contiguous()
        grad = SPLUSolveLayer.solve(ctx.solver, grad_output)
        # At this point we perform a NAN check because the backsolve sometimes returns NaNs.
        assert not torch.isnan(grad).any(), (
            "Nan in the backward pass of the POISSON SOLVE"
        )
        return None, grad

    @staticmethod
    def solve(
        solver: CholeskySolverD, b: Float[Tensor, "B V-1 3"]
    ) -> Float[Tensor, "B V-1 3"]:
        """Solves the linear system defined by the solver for a given rhs.

        Args:
            solver: the cholesky solver of the (reduced) poisson system.
            b: the right hand side to solve for; if it is a matrix with
                multiple columns, a solution is computed for each column.

        Returns:
            Solution x which satisfies A x = b, where A is the poisson
            system solver describes.
        """
        if b.device.type == "cpu":
            assert b.shape[0] == 1, "Need to code parrallel implem on the first dim"
            b = b.squeeze()
            b_cpu = b.double().cpu()
            x = torch.zeros_like(b_cpu)
            solver.solve(b_cpu, x)
            return x.contiguous().to(b.device).unsqueeze(0)
        else:
            b = b.double().contiguous()
            c = b.permute(1, 2, 0).contiguous()
            c = c.view(c.shape[0], -1)
            x = torch.zeros_like(c)
            solver.solve(c, x)
            x = x.view(b.shape[1], b.shape[2], b.shape[0])
            x = x.permute(2, 0, 1).contiguous()
            return x.contiguous()


def _predicted_jacobians_to_vertices_via_poisson_solve(
    L: CholeskySolverD,
    rhs: Float[Tensor, "V-1 3F"],
    jacobians: Float[Tensor, "B 3F 3"],
) -> Float[Tensor, "B V 3"]:
    """Converts predicted jacobians to the poisson convention and solves.

    Args:
        L: the cholesky solver of the (reduced) poisson system.
        rhs: the (reduced) rhs of the poisson system.
        jacobians: flat per-face jacobian rows in the interleaved
            (3f + c) convention.

    Returns:
        The reconstructed vertex positions (with a zero row prepended and
        mean-centered), one per batch item.
    """

    def _batch_rearrange_input(
        input: Float[Tensor, "B 3F 3"],
    ) -> Float[Tensor, "B 3F 3"]:
        """Rearranges flat rows from (3f + c) to component-major blocks.

        Args:
            input: flat jacobian rows in the (3f + c) convention.

        Returns:
            The same values with rows grouped into three consecutive
            per-component blocks.
        """
        assert isinstance(input, torch.Tensor) and len(input.shape) in [2, 3]
        P = torch.zeros(input.shape).type_as(input)
        if len(input.shape) == 3:
            # Batched input
            k = input.shape[1] // 3
            P[:, :k, :] = input[:, ::3]
            P[:, k : 2 * k, :] = input[:, 1::3]
            P[:, 2 * k :, :] = input[:, 2::3]

        else:
            k = input.shape[0] // 3
            P[:k, :] = input[::3]
            P[k : 2 * k, :] = input[1::3]
            P[2 * k :, :] = input[2::3]

        return P

    P = _batch_rearrange_input(jacobians)
    assert isinstance(P, torch.Tensor) and len(P.shape) in [2, 3]
    assert len(P.shape) == 3
    P = P.double()
    input_to_solve = _multiply_sparse_2d_by_dense_3d(rhs, P)
    out = SPLUSolveLayer.apply(L, input_to_solve)
    out = torch.cat(
        [torch.zeros(out.shape[0], 1, out.shape[2]).type_as(out), out], dim=1
    )  ## Why?? Because!
    out = out - torch.mean(out, axis=1, keepdim=True)
    return out.type_as(jacobians)


def _multiply_sparse_2d_by_dense_3d(
    mat: Float[Tensor, "3F V"],
    B: Float[Tensor, "B V K"],
) -> Float[Tensor, "B 3F K"]:
    """Multiplies a sparse 2D tensor with a batch of dense 2D tensors.

    Args:
        mat: sparse COO tensor, converted to CSR internally.
        B: batch of dense tensors.

    Returns:
        The batched product of `mat` with each item of `B`.
    """
    ret = []
    csr = mat.to_sparse_csr().to(B.dtype)
    for i in range(B.shape[0]):
        C = torch.sparse.mm(csr, B[i, ...])
        ret.append(C)
    ret = torch.stack(tuple(ret))
    return ret


def _coo_to_cholesky(
    coo: Float[Tensor, "V-1 V-1"],
) -> CholeskySolverD:
    """Builds a cholespy solver from a sparse COO tensor.

    Args:
        coo: sparse COO representation of the system matrix.

    Returns:
        A cholespy solver that can solve the system on CPU or GPU.
    """
    coo = coo.coalesce()
    indices = coo.indices()
    return CholeskySolverD(
        coo.shape[0],
        indices[0],
        indices[1],
        coo.values(),
        MatrixType.COO,
    )


def gradient_operator(
    V: Float[Tensor, "V 3"],
    F: Integer[Tensor, "F 3"],
) -> Float[Tensor, "3F V"]:
    """Computes the interleaved gradient operator of the mesh.

    Args:
        V: vertex positions.
        F: face indices.

    Returns:
        G: sparse COO tensor of shape (3M, N):

            For each face f:
                G[3*f + 0] = x component of gradient
                G[3*f + 1] = y component of gradient
                G[3*f + 2] = z component of gradient

            Thus:
                grad = G @ vertex_scalars
                grad.reshape(M, 3) -> per-face gradients
    """
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
    # (M, 3 vertices, 3 xyz components)
    values = torch.stack([grad_phi0, grad_phi1, grad_phi2], dim=1)
    M = F.shape[0]
    # Each face has 3 rows: x, y, z; each row references the 3 face vertices
    row_idx = (
        3 * torch.arange(M, device=F.device)[:, None]
        + torch.arange(3, device=F.device)[None, :]
    )[..., None].expand(-1, -1, 3)
    col_idx = F[:, None, :].expand(-1, 3, -1)
    indices = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)])
    # values needs to be (M, xyz, vertex), matching rows/columns
    values = values.transpose(1, 2)
    G = torch.sparse_coo_tensor(
        indices,
        values.reshape(-1),
        size=(3 * M, V.shape[0]),
        device=V.device,
        dtype=V.dtype,
    )
    return G.coalesce()


def tangential_basis(
    V: Float[Tensor, "V 3"],
    F: Integer[Tensor, "F 3"],
) -> Float[Tensor, "F 3 2"]:
    """Computes orthonormal tangent bases for each face.

    Args:
        V: vertex positions.
        F: face indices.

    Returns:
        basis: (F, 3, 2) two orthonormal tangent vectors per face.
    """
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]

    e1 = p1 - p0
    e2 = p2 - p0

    t1 = torch_F.normalize(e1, dim=-1)
    n = torch_F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)

    return torch.stack((t1, t2), dim=-1)


def _stacked_double_areas(
    V: Float[Tensor, "V 3"],
    F: Integer[Tensor, "F 3"],
) -> Float[Tensor, "3F"]:
    """Computes per-face double areas with interleaved repetition.

    Args:
        V: vertex positions.
        F: face indices.

    Returns:
        d_area: (3F,) tensor with each face's doubled area repeated for
        its x, y and z gradient rows (matching the row interleaving of
        gradient_operator).
    """
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    d_area = torch.linalg.norm(torch.cross(p1 - p0, p2 - p0, dim=-1), dim=-1)
    return d_area.repeat_interleave(3)


def _torch_coo_to_scipy(
    mtx: Float[Tensor, "A B"],
) -> scipy.sparse.coo_matrix:
    """Converts a torch sparse COO tensor to a scipy coo matrix.

    Args:
        mtx: sparse COO torch tensor.

    Returns:
        The same matrix as a scipy sparse coo matrix.
    """
    mtx = mtx.coalesce()
    idx = mtx.indices().cpu().numpy()
    return scipy.sparse.coo_matrix(
        (mtx.values().cpu().numpy(), (idx[0], idx[1])), shape=tuple(mtx.shape)
    )


def _scipy_coo_to_torch(
    mtx: scipy.sparse.coo_matrix,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "A B"]:
    """Converts a scipy coo matrix to a torch sparse COO tensor.

    Args:
        mtx: scipy sparse coo matrix.
        device: device of the resulting tensor.
        dtype: dtype of the resulting tensor.

    Returns:
        The same matrix as a torch sparse COO tensor.
    """
    indices = numpy.vstack((mtx.row, mtx.col))
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(mtx.data).type(dtype),
        size=mtx.shape,
        device=device,
        dtype=dtype,
    ).coalesce()


def remove_first_coo(
    mtx: Float[Tensor, "A B"],
    row: bool = True,
    col: bool = True,
) -> Float[Tensor, "A-1 B-1"]:
    """Removes the first row and/or column of a sparse COO tensor.

    The remaining entries are re-indexed. Used to pin the constant
    null-space of the Laplacian at vertex 0 (see
    PoissonSystem.from_mesh for the full pinning convention).

    Args:
        mtx: sparse COO torch tensor.
        row: whether to remove and re-index the first row.
        col: whether to remove and re-index the first column.

    Returns:
        The reduced sparse COO tensor.
    """
    mtx = mtx.coalesce()
    idx = mtx.indices()
    val = mtx.values()
    mask = torch.ones(idx.shape[1], dtype=torch.bool, device=idx.device)
    if row:
        mask &= idx[0] > 0
    if col:
        mask &= idx[1] > 0
    idx = idx[:, mask]
    if row:
        idx[0] -= 1
    if col:
        idx[1] -= 1
    shape = list(mtx.shape)
    if row:
        shape[0] -= 1
    if col:
        shape[1] -= 1
    return torch.sparse_coo_tensor(
        idx,
        val[mask],
        size=shape,
        device=mtx.device,
        dtype=mtx.dtype,
    ).coalesce()
