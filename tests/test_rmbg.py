import torch

from vdsds.utils.rmbg import BackgroundRemover


class _StubModel:
    """Foreground matte: 1 on the upper half, 0 below."""

    def __call__(self, x):
        assert x.shape[1] == 3
        matte = torch.zeros(1, 1, x.shape[-2], x.shape[-1])
        matte[:, :, : x.shape[-2] // 2, :] = 1.0
        return [matte]


def test_remove_plumbing(monkeypatch):
    """RGB passes through; the stubbed matte becomes the alpha channel
    (resampled back to the input resolution)."""
    monkeypatch.setattr(
        BackgroundRemover, "_load_model", classmethod(lambda cls, device=None: _StubModel())
    )
    img = torch.rand(2, 3, 40, 60)
    out = BackgroundRemover.remove(img, device="cpu")
    assert out.shape == (2, 4, 40, 60)
    assert torch.equal(out[:, :3], img)
    alpha = out[:, 3]
    # 512/1024 of the height -> upper rows matte 1, lower rows 0
    # (kept off the bilinear boundary row; float-tolerant).
    assert torch.all(alpha[:, :15] > 0.999)
    assert torch.all(alpha[:, 30:] < 0.001)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_remove_gray_input_expanded(monkeypatch):
    monkeypatch.setattr(
        BackgroundRemover, "_load_model", classmethod(lambda cls, device=None: _StubModel())
    )
    gray = torch.rand(1, 1, 40, 60)
    out = BackgroundRemover.remove(gray, device="cpu")
    assert out.shape == (1, 4, 40, 60)
    # Gray is expanded to RGB: all three channels equal the input.
    assert torch.equal(out[:, 0], gray[:, 0])
    assert torch.equal(out[:, 1], gray[:, 0])
    assert torch.equal(out[:, 2], gray[:, 0])


def test_remove_input_type_validation(monkeypatch):
    monkeypatch.setattr(
        BackgroundRemover, "_load_model", classmethod(lambda cls, device=None: _StubModel())
    )
    with __import__("pytest").raises(AssertionError, match="BCHW"):
        BackgroundRemover.remove(torch.rand(3, 40, 60), device="cpu")


def test_default_device_follows_cuda():
    device = BackgroundRemover.default_device()
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")
