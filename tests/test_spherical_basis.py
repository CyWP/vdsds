import pytest
import torch

from vdsds.utils.spherical_basis import SphericalGaussianBasis


@pytest.fixture
def basis():
    torch.manual_seed(0)
    b = SphericalGaussianBasis(num_funcs=6, num_dims=6, batch_size=3)
    b.weights.requires_grad_(True)
    b.log_sigmas.requires_grad_(True)
    b.centroids.requires_grad_(True)
    return b


def test_from_cartesian_at_exact_pole_finite(basis):
    vec = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    out = basis.from_cartesian(vec)
    assert torch.isfinite(out).all()


def test_from_cartesian_gradients_finite_at_pole(basis):
    # Put one axis exactly at the pole and query there exactly.
    basis.centroids.data[0, 0] = torch.tensor([0.0, 0.0, 1.0])
    vec = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    out = basis.from_cartesian(vec)
    loss = out.abs().sum()
    loss.backward()
    for name in ["weights", "log_sigmas", "centroids"]:
        assert torch.isfinite(getattr(basis, name).grad).all(), name
    assert torch.isfinite(vec.grad).all()


def test_collapsed_width_finite(basis):
    basis.log_sigmas.data.fill_(-30.0)  # near-zero width
    # Query exactly on an axis (g == 0 with tiny width: 0/0 risk)
    basis.centroids.data[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    vec = torch.tensor([[1.0, 0.0, 0.0]])
    out = basis.from_cartesian(vec)
    assert torch.isfinite(out).all()
    (out.abs().sum()).backward()
    assert torch.isfinite(basis.log_sigmas.grad).all()
    assert torch.isfinite(basis.centroids.grad).all()


def test_smoothness_loss_finite_and_grads(basis):
    basis.log_sigmas.data.fill_(-30.0)
    out = basis.smoothness_loss()
    assert torch.isfinite(out).all()
    out.backward()
    assert torch.isfinite(basis.log_sigmas.grad).all()
    assert torch.isfinite(basis.weights.grad).all()
    assert torch.isfinite(basis.centroids.grad).all()


def test_dot_falloff_matches_radian_semantics(basis):
    # With the default init (sigma_dot = sigma_r^2 / 2) the falloff matches
    # the old exp(-theta^2 / (2 sigma_r^2)): exp(-1/2) at the nominal
    # radius sigma_r and exp(-1) at its 1/e radius sqrt(2)*sigma_r.
    import math

    sigma_r = math.sqrt(2 * 0.5 / 6)
    basis.centroids.data[0, 0] = torch.tensor([0.0, 0.0, 1.0])
    basis.weights.data.zero_()
    basis.weights.data[0, 0, 0] = 1.0

    def value_at(theta):
        x, z = torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))
        vec = torch.tensor([[x.item(), 0.0, z.item()]])
        return basis.from_cartesian(vec)[0, 0, 0]

    # exp(-1/2) to within the small-angle approximation error
    # (1 - cos(theta) ~ theta^2/2 has ~1% deviation at theta ~ 0.4 rad).
    assert torch.allclose(
        value_at(sigma_r),
        torch.exp(torch.tensor(-0.5)),
        rtol=1e-2,
    )
    assert torch.allclose(
        value_at(sigma_r * math.sqrt(2)),
        torch.full((), 1 / math.e, dtype=basis.centroids.dtype),
        rtol=4e-2,
    )


def test_gradcheck_smooth_at_axis_and_antipode():
    torch.manual_seed(1)
    b = SphericalGaussianBasis(num_funcs=3, num_dims=2, batch_size=1)
    u = torch.nn.functional.normalize(
        torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, -1.0, -0.1], [1.0, 2.0, 3.0]]]
        ),
        dim=-1,
    )

    def f(u, centroids, log_sigmas, weights):
        b.centroids = torch.nn.Parameter(centroids)
        b.log_sigmas = torch.nn.Parameter(log_sigmas)
        b.weights = torch.nn.Parameter(weights)
        return (b._from_unit_directions(u) ** 2).sum()

    assert torch.autograd.gradcheck(
        f,
        (
            u,
            b.centroids.detach().clone().requires_grad_(True),
            b.log_sigmas.detach().clone().requires_grad_(True),
            b.weights.detach().clone().requires_grad_(True),
        ),
        eps=1e-5,
        atol=1e-5,
        rtol=1e-3,
        check_undefined_grad=False,
    )
