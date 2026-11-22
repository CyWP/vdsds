import torch
import torch.nn.functional as F

from vdsds.utils.deepfloyd import DeepFloydGuidance


class _StubScheduler:
    """add_noise with a decreasing alphas_cumprod, standard formula."""

    def __init__(self, n: int = 1000):
        self.alphas_cumprod = torch.linspace(0.9999, 0.01, n)

    def add_noise(self, latents, noise, t):
        a = self.alphas_cumprod[t].view(-1, 1, 1, 1).sqrt()
        return a * latents + (1 - a**2).sqrt() * noise


def _stub_guidance():
    """Bare DeepFloydGuidance with a linear cond score: eps = 0.5*x + 0.1.

    The stub records every (latents, t, embeds) call for pairing/
    band assertions.
    """
    df = DeepFloydGuidance.__new__(DeepFloydGuidance)
    df.device = torch.device("cpu")
    df.scheduler = _StubScheduler()
    df.min_step, df.max_step = 10, 900
    df.alphas = df.scheduler.alphas_cumprod
    calls = []

    def fake_unet(latents, t, embeds):
        calls.append((latents.detach().clone(), t.clone(), embeds.shape))
        return latents * 0.5 + 0.1

    df.forward_unet = fake_unet
    return df, calls


def test_dds_paired_noise_and_gradient():
    """Both branches share (t, noise); the gradient is the paired
    cond-score difference weighted by w_t."""
    df, calls = _stub_guidance()
    torch.manual_seed(0)
    render = torch.rand(1, 3, 32, 32)
    ref = torch.rand(1, 3, 32, 32)
    embed = torch.rand(4, 8)
    t = torch.tensor([200])
    noise = torch.zeros(1, 3, 64, 64)
    out = df.DDS(render, ref, embed, t=t, noise=noise)
    lat_r = F.interpolate(render * 2 - 1, (64, 64), mode="bilinear", align_corners=False)
    lat_f = F.interpolate(ref * 2 - 1, (64, 64), mode="bilinear", align_corners=False)
    # eps = 0.5 * x_t + 0.1, x_t = sqrt(abar) * latents (zero noise):
    # grad = w_t * (eps_ren - eps_ref) = w_t * 0.5 * sqrt(abar) * dlat.
    a = df.alphas[t].view(-1, 1, 1, 1).sqrt()
    expected = (1 - df.alphas[t]).view(-1, 1, 1, 1) * 0.5 * a * (lat_r - lat_f)
    assert torch.allclose(out["grad"], expected, atol=1e-5)
    assert torch.isfinite(out["loss_sds"]) and out["loss_sds"] >= 0.0
    # One batched call, two rows (render + reference): cond branch only.
    assert len(calls) == 1
    lats, ts, embed_shape = calls[0]
    assert lats.shape[0] == 2 and embed_shape == (2, 4, 8)
    # Paired: identical t across both branch rows.
    assert torch.equal(ts[0], ts[1])
    # Paired: identical noise -- the two noised rows differ only by
    # sqrt(abar) * (lat_r - lat_f).
    diff = lats[0] - lats[1]
    assert torch.allclose(
        diff, a[0] * (lat_r[0] - lat_f[0]), atol=1e-5
    )


def test_dds_timestep_band():
    """With t=None, timesteps are sampled inside the lower t_max band."""
    df, calls = _stub_guidance()
    render = torch.rand(1, 3, 32, 32)
    ref = torch.rand(1, 3, 32, 32)
    for t_max in (0.25, 0.5):
        calls.clear()
        for _ in range(10):
            df.DDS(render, ref, torch.rand(4, 8), t_max=t_max)
        ts = torch.cat([c[1] for c in calls])
        hi = int(df.min_step + t_max * (df.max_step - df.min_step))
        assert (ts >= df.min_step).all() and (ts < hi).all()


def test_dds_embeds_expanded_per_batch():
    """A (T, D) embedding is broadcast to both branch rows."""
    df, calls = _stub_guidance()
    render = torch.rand(2, 3, 32, 32)
    ref = torch.rand(2, 3, 32, 32)
    df.DDS(render, ref, torch.ones(4, 8), t=torch.tensor([10, 20]))
    _, _, embed_shape = calls[0]
    assert embed_shape == (4, 4, 8)  # (2 branches x B=2, T, D)


def test_dds_shape_mismatch_raises():
    df, _ = _stub_guidance()
    with __import__("pytest").raises(AssertionError, match="mismatch"):
        df.DDS(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 48, 48), torch.ones(4, 8))
