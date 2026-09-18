import os
from pathlib import Path

import pytest
import torch

from vdsds.utils.poisson_system import PoissonSystem
from vdsds.utils.poisson_system_bak import poisson_system_matrices_from_mesh

pytest.importorskip("igl")
pytest.importorskip("torch_sparse")

COW_PATH = os.path.expanduser("~/shared/3D/models/cow.glb")

# Note: numerics between the igl based pipeline and this pure-torch one agree
# up to small area-normalization differences (~1e-3 relative). We therefore
# assert "same in spirit": matching dims/permutations with loose relative
# tolerances, plus exact self-consistency checks on the njf solver itself.
RTOL = 1e-2
ATOL = 1e-6


def _grid_mesh(n=6):
    xs = torch.linspace(0.0, 1.0, n)
    ys = torch.linspace(0.0, 1.0, n)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    V = torch.stack(
        [gx.reshape(-1), gy.reshape(-1), torch.sin(gx.reshape(-1) * 3.0) * 0.05],
        dim=-1,
    )
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            F.append([a, c, d])
            F.append([a, d, b])
    return V.double(), torch.tensor(F, dtype=torch.int64)


@pytest.fixture(scope="module")
def systems():
    V, F = _grid_mesh()
    psm = poisson_system_matrices_from_mesh(
        V.numpy(), F.numpy(), ttype=torch.float64, cpuonly=True
    )
    solver = psm.create_poisson_solver()
    njf_system = PoissonSystem.from_mesh(V, F)
    return V, solver, njf_system


def _dense(mtx):
    if hasattr(mtx, "to_coo"):
        return torch.from_numpy(mtx.to_coo().toarray())
    return mtx.to_dense()


def test_shapes_match(systems):
    _, solver, ps = systems
    nf = ps.W.shape[0]
    N = ps.grad.shape[1]
    assert ps.grad.shape == (3 * nf, N)
    assert ps.L.shape == (N - 1, N - 1)
    assert ps.rhs.shape == (N - 1, 3 * nf)
    assert solver.lap.n == N - 1 and solver.lap.m == N - 1
    assert solver.rhs.n == N - 1 and solver.rhs.m == 3 * nf


def test_laplacian_matches(systems):
    _, solver, ps = systems
    L_old, L_njf = _dense(solver.lap), ps.L.to_dense()
    max_gap = (L_old - L_njf).abs().max()
    scale = L_old.abs().max()
    assert max_gap / scale < RTOL, f"lap rel diff {max_gap / scale}"


def test_rhs_matches(systems):
    _, solver, ps = systems
    R_old, R_njf = _dense(solver.rhs), ps.rhs.to_dense()
    assert R_old.shape == R_njf.shape
    max_gap = (R_old - R_njf).abs().max()
    scale = R_old.abs().max()
    assert max_gap / scale < RTOL, f"rhs rel diff {max_gap / scale}"


def test_grad_matches(systems):
    _, solver, ps = systems
    G_njf = ps.grad.to_dense()
    G_old = solver.sparse_grad.multiply_with_dense(
        torch.eye(ps.grad.shape[1], dtype=torch.float64)
    )
    assert G_old.shape == G_njf.shape
    assert G_old.abs().max() / G_njf.abs().max() < 1.0 + RTOL
    rel = (G_old - G_njf).abs().max() / G_njf.abs().max()
    assert rel < RTOL, f"grad rel diff {rel}"


def test_w_matches(systems):
    _, solver, ps = systems
    W_old, W_njf = solver.W, ps.W
    same = torch.allclose(W_old, W_njf, atol=1e-10)
    flipped = torch.allclose(W_old, W_njf * torch.tensor([1.0, -1.0]), atol=1e-10)
    assert same or flipped, "W differs from igl.local_basis beyond a sign flip"


def test_jacobians_from_vertices_match(systems):
    V, solver, ps = systems
    J_old = solver.jacobians_from_vertices(V.unsqueeze(0))
    J_njf = ps.jacobians_from_vertices(V.unsqueeze(0))
    rel = (J_old - J_njf).abs().max() / J_njf.abs().max()
    assert rel < RTOL, f"jacobians rel diff {rel}"


def test_restricted_jacobians_match(systems):
    V, solver, ps = systems
    JR_old = solver.restricted_jacobians_from_vertices(V.unsqueeze(0))
    JR_njf = ps.restricted_jacobians_from_vertices(V.unsqueeze(0))
    rel = (JR_old - JR_njf).abs().max() / JR_njf.abs().max()
    assert rel < RTOL, f"restricted jacobians rel diff {rel}"


@pytest.fixture
def cpu_cholespy(monkeypatch):
    import vdsds.utils.poisson_system_bak as psm_mod

    monkeypatch.setattr(psm_mod, "USE_CHOLESPY_GPU", False)
    monkeypatch.setattr(psm_mod, "USE_CHOLESPY_CPU", True)


def test_solve_poisson_matches(systems, cpu_cholespy):
    _, solver, ps = systems
    torch.manual_seed(0)
    nf = ps.W.shape[0]
    J = torch.rand(1, nf, 3, 3, dtype=torch.float64) * 0.1 + torch.eye(3)

    sol_old = solver.solve_poisson(J)
    sol_njf = ps.solve_poisson(J)
    rel = (sol_old - sol_njf).abs().max() / sol_njf.abs().max()
    assert rel < RTOL, f"solve rel diff {rel}"
    assert torch.allclose(
        (sol_old - sol_old.mean(dim=1, keepdim=True)).norm(dim=-1),
        (sol_njf - sol_njf.mean(dim=1, keepdim=True)).norm(dim=-1),
        rtol=RTOL,
    )


def test_solve_poisson_reconstructs_vertices(systems):
    V, _, ps = systems
    J = ps.jacobians_from_vertices(V.unsqueeze(0))
    sol = ps.solve_poisson(J)[0]
    V0 = V - V.mean(dim=0)
    assert torch.allclose(sol, V0, atol=1e-8), "njf identity solve must reconstruct V"


def _cow_mesh():
    if not os.path.exists(COW_PATH):
        pytest.skip("cow.glb not available")
    from vdsds.utils.loading import load_glb

    data = load_glb(Path(COW_PATH))
    return data["V"], data["F"]


@pytest.fixture(scope="module")
def cow_systems():
    V, F = _cow_mesh()
    V = V.double()
    psm = poisson_system_matrices_from_mesh(
        V.numpy(), F.numpy(), ttype=torch.float64, cpuonly=True
    )
    solver = psm.create_poisson_solver()
    ps = PoissonSystem.from_mesh(V, F)
    import vdsds.utils.poisson_system_bak as psm_mod

    psm_mod.USE_CHOLESPY_GPU = False
    psm_mod.USE_CHOLESPY_CPU = True
    return V, solver, ps


def test_cow_self_reconstructs(cow_systems):
    V, _, ps = cow_systems
    assert V.shape[1] == 3 and ps.W.shape[0] > 0
    J = ps.jacobians_from_vertices(V.unsqueeze(0))
    sol = ps.solve_poisson(J)[0]
    assert sol.shape == V.shape
    V0 = V - V.mean(dim=0)
    err = (sol - V0).abs().max() / (V0.abs().max() + 1e-12)
    assert err < 1e-8, f"cow reconstruction rel error {err}"


def test_cow_solves_match(cow_systems):
    V, solver, ps = cow_systems
    torch.manual_seed(1)
    J = torch.rand(1, ps.W.shape[0], 3, 3, dtype=torch.float64) * 0.05 + torch.eye(3)
    sol_old = solver.solve_poisson(J)
    sol_njf = ps.solve_poisson(J)
    rel = (sol_old - sol_njf).norm(dim=-1) / (sol_njf.norm(dim=-1) + 1e-12)
    # Random per-face jacobians are very high frequency; the two pipelines
    # agree up to the area-normalization detail, so only require that both
    # produce the same magnitude/shape solution, with loose relative error.
    assert rel.max() < 0.2, f"cow solve rel diff {rel.max()}"


def test_cow_jacobians_match(cow_systems):
    V, solver, ps = cow_systems
    J_old = solver.jacobians_from_vertices(V.unsqueeze(0))
    J_njf = ps.jacobians_from_vertices(V.unsqueeze(0))
    assert J_old.shape == J_njf.shape
    rel = (J_old - J_njf).abs().max() / J_njf.abs().max()
    assert rel < RTOL, f"cow jacobians rel diff {rel}"


def test_solve_poisson_gradcheck():
    V, F = _grid_mesh(n=4)
    ps = PoissonSystem.from_mesh(V, F)
    delta = torch.zeros_like(V, requires_grad=True)

    def f(d):
        return ps.solve_poisson(ps.jacobians_from_vertices((V + d).unsqueeze(0)))[0]

    assert torch.autograd.gradcheck(f, (delta,), eps=1e-5, atol=1e-6, rtol=1e-4)


def test_permuted_faces():
    V, F = _grid_mesh(n=6)
    perm_gen = torch.Generator().manual_seed(3)
    perm = torch.randperm(F.shape[0], generator=perm_gen)
    flip = torch.arange(F.shape[0]) % 3 == 0
    Fp = F[perm]
    rev = Fp[:, [2, 1, 0]]
    Fp = torch.where(flip[:, None], rev, Fp)
    ps_p = PoissonSystem.from_mesh(V, Fp)
    psm = poisson_system_matrices_from_mesh(
        V.numpy().copy(), Fp.numpy().copy(), ttype=torch.float64, cpuonly=True
    )
    solver_p = psm.create_poisson_solver()
    import vdsds.utils.poisson_system_bak as psm_mod

    psm_mod.USE_CHOLESPY_GPU = False
    psm_mod.USE_CHOLESPY_CPU = True

    Vb = V.unsqueeze(0)
    J = ps_p.jacobians_from_vertices(Vb)
    sol = ps_p.solve_poisson(J)[0]
    V0 = V - V.mean(dim=0)
    err = (sol - V0).abs().max() / V0.abs().max()
    assert err < 1e-8, f"permuted reconstruction rel error {err}"

    J_old = solver_p.jacobians_from_vertices(Vb)
    J_njf = ps_p.jacobians_from_vertices(Vb)
    rel = (J_old - J_njf).abs().max() / J_njf.abs().max()
    assert rel < RTOL, f"permuted jacobians rel diff {rel}"


def test_tangent_expansion_roundtrip(systems):
    V, _, ps = systems
    nf = ps.W.shape[0]
    torch.manual_seed(0)
    D = torch.rand(1, nf, 3, 3, dtype=torch.float64)
    R = ps.restrict_jacobians(D)
    J_full = ps.expand_tangent_jacobians(R)
    assert torch.allclose(R, J_full @ ps.W, atol=1e-12)


def test_zero_tangent_deformation_reconstructs(systems):
    V, _, ps = systems
    nf = ps.W.shape[0]
    J_src = ps.jacobians_from_vertices(V.unsqueeze(0))
    zeros_tan = torch.zeros(1, nf, 3, 2, dtype=torch.float64)
    J_disp = (
        torch.eye(3, dtype=torch.float64)[None]
        + torch.einsum("bfae,fde->bfad", zeros_tan, ps.W)
    )
    J_transformed = torch.einsum("bfij,bfjk->bfik", J_disp, J_src)
    sol = ps.solve_poisson(J_transformed)[0]
    V0 = V - V.mean(dim=0)
    assert torch.allclose(sol, V0, atol=1e-8)


def test_random_tangent_deformation_is_finite(systems):
    V, _, ps = systems
    nf = ps.W.shape[0]
    torch.manual_seed(2)
    J_src = ps.jacobians_from_vertices(V.unsqueeze(0))
    for scale in [0.0, 1.0, 10.0]:
        J_tan = torch.rand(1, nf, 3, 2, dtype=torch.float64) * scale
        J_disp = torch.eye(3, dtype=torch.float64)[None] + ps.expand_tangent_jacobians(
            J_tan
        )
        J_transformed = torch.einsum("bfij,bfjk->bfik", J_disp, J_src)
        sol = ps.solve_poisson(J_transformed)
        assert torch.isfinite(sol).all(), f"non-finite solution at scale {scale}"
