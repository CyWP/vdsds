import importlib.util

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor

from vdsds.rendering.shader import NO_DEPTH
from vdsds.utils.depth import DepthAnything, depth_loss


class _FakePrediction:
    def __init__(self, depth: np.ndarray):
        self.depth = depth  # (N, H, W) like the real Prediction


class _Recorder:
    """Tracks load calls and inference inputs across fresh fake models."""

    def __init__(self):
        self.loads = []
        self.inputs = []  # one entry (a 1-list of inputs) per inference
        self.depth_single = (
            np.linspace(0.0, 1.0, 60 * 80).reshape(60, 80).astype(np.float32)
        )


class _FakeModel:
    """Stand-in for depth_anything_3.api.DepthAnything3 instances."""

    def __init__(self, rec: _Recorder):
        self.rec = rec

    def inference(self, image, **kwargs):
        self.rec.inputs.append(image)
        stack = np.stack([self.rec.depth_single] * len(image), axis=0)
        return _FakePrediction(stack)

    def to(self, device):
        return self


@pytest.fixture
def fake_model(monkeypatch):
    """Fresh-model semantics: every _load_model call returns a new fake
    (the wrapper must never cache the model across calls)."""
    rec = _Recorder()

    def fake_load(cls, device=None):
        rec.loads.append(device)
        return _FakeModel(rec)

    monkeypatch.setattr(DepthAnything, "_load_model", classmethod(fake_load))
    return rec


# ---------------------------------------------------------------- plumbing


def test_input_rgb_tensor_converted_to_pil(fake_model):
    imgs = torch.zeros(1, 3, 16, 32)
    imgs[0, 0, 8, 16] = 1.0  # red pixel
    DepthAnything.depth(imgs)
    pil = fake_model.inputs[0][0]
    assert isinstance(pil, Image.Image)
    assert pil.size == (32, 16)
    assert pil.mode == "RGB"
    assert pil.getpixel((16, 8)) == (255, 0, 0)


def test_input_rgba_drops_alpha(fake_model):
    imgs = torch.zeros(1, 4, 16, 32)
    imgs[0, 0, 8, 16] = 1.0  # red pixel with alpha 1
    DepthAnything.depth(imgs)
    assert fake_model.inputs[0][0].getpixel((16, 8)) == (255, 0, 0)


def test_input_gray_expanded_to_rgb(fake_model):
    DepthAnything.depth(torch.ones(1, 1, 12, 12))
    pil = fake_model.inputs[0][0]
    assert pil.mode == "RGB"
    assert pil.size == (12, 12)
    assert pil.getpixel((6, 6)) == (255, 255, 255)


def test_batch_per_image_inputs_single_load(fake_model):
    imgs = torch.rand(3, 3, 12, 12)
    depth = DepthAnything.depth(imgs)
    # Per-image inference within a single load/unload cycle.
    assert len(fake_model.inputs) == 3
    assert len(fake_model.loads) == 1
    assert isinstance(fake_model.inputs[0][0], Image.Image)
    assert depth.shape == (3, 1, 60, 80)


def test_non_tensor_input_rejected(fake_model):
    with pytest.raises(AssertionError):
        DepthAnything.depth(torch.rand(12, 12, 3))  # HWC, not BCHW


def test_invalid_ndim_rejected(fake_model):
    with pytest.raises(AssertionError):
        DepthAnything.depth(torch.rand(12, 12))


def test_infer_depths_set_api(fake_model):
    imgs = [torch.rand(3, 12, 12), torch.rand(4, 20, 20)]
    depths = DepthAnything.infer_depths(imgs)
    assert len(depths) == 2
    for d in depths:
        assert d.shape == (60, 80)
        assert d.dtype == torch.float32
    assert len(fake_model.loads) == 1  # one load/unload cycle for the set


# ----------------------------------------------------------------- output


def test_depth_output_shape_and_dtype(fake_model):
    depth = DepthAnything.depth(torch.rand(1, 3, 16, 32))
    assert isinstance(depth, Tensor)
    assert depth.shape == (1, 1, 60, 80)  # BCHW, model-processed resolution
    assert depth.dtype == torch.float32
    assert depth.device.type == "cpu"
    expected = torch.from_numpy(fake_model.depth_single)
    assert torch.allclose(depth[0, 0], expected)


def test_model_loaded_per_call_never_cached(fake_model):
    DepthAnything.depth(torch.rand(1, 3, 8, 8))
    DepthAnything.depth(torch.rand(1, 3, 8, 8))
    # One fresh load per inference call: no model persists on the class.
    assert len(fake_model.loads) == 2
    assert not hasattr(DepthAnything, "_model")


# ------------------------------------------------------------- depth loss


def _masked_depth(values=None, shape=(1, 1, 8, 8), bg=NO_DEPTH):
    """Build a ``(1, 1, 8, 8)`` map with a valid 6x6 foreground block
    (placeholder 0.5s, overwritten by ``_raster_ramp`` for ramps),
    ``bg``-sentinel in the 1px border, and explicit per-pixel values
    applied last."""
    map = torch.full(shape, bg, dtype=torch.float32)
    for y in range(1, 7):
        for x in range(1, 7):
            map[0, 0, y, x] = 0.5
    for (y, x), v in (values or {}).items():
        map[0, 0, y, x] = v
    return map


def _raster_ramp(img, start=1e-3, stop=1.0):
    """Fill the valid (non-sentinel) pixels of ``img[..., 0, ...]`` with a
    deterministic linear ramp in (0, 1], preserving the background."""
    valid = img[0, 0] > NO_DEPTH
    idx = valid.nonzero()
    ramp = torch.linspace(start, stop, idx.shape[0])
    for i, (y, x) in enumerate(idx.tolist()):
        img[0, 0, y, x] = ramp[i]
    return img


def test_no_depth_sentinel_shared():
    assert DepthAnything.no_depth == NO_DEPTH == -1.0


def test_affine_loss_recovers_exact_scaling():
    rendered = _raster_ramp(_masked_depth({}))
    measured = 2.5 * rendered + 0.3  # exact affine link on valid pixels
    loss = depth_loss(rendered, measured)
    assert loss.item() < 1e-5


def test_affine_loss_handles_inverted_ordering():
    # Monocular relative depth often encodes nearness, clip-space depth
    # distance; the closed-form fit may flip sign.
    rendered = _raster_ramp(_masked_depth({}))
    measured = 1.0 - rendered
    loss = depth_loss(rendered, measured)
    assert loss.item() < 1e-5


def test_loss_ignores_background_pixels():
    rendered = _raster_ramp(_masked_depth({}))
    base = 2.5 * rendered + 0.3
    with_garbage = base.clone()
    with_garbage[0, 0][rendered[0, 0] == NO_DEPTH] = 1e6
    zero_bg = base.clone()
    zero_bg[0, 0][rendered[0, 0] == NO_DEPTH] = 0.0
    l1 = depth_loss(rendered, with_garbage)
    l2 = depth_loss(rendered, zero_bg)
    assert torch.allclose(l1, l2)


def test_loss_counts_any_negative_value_as_no_depth():
    rendered = _raster_ramp(_masked_depth({}))
    # Same map, but one background pixel holds a non-sentinel negative.
    neg_bg = rendered.clone()
    neg_bg[0, 0, 7, 7] = -0.37
    measured = 2.5 * rendered + 0.3  # valid-block link...
    measured[0, 0, 7, 7] = 1e6  # ...with junk at that background pixel
    # Below-zero masking must make both variants identical: matching
    # only the exact sentinel would have treated -0.37 as valid
    # geometry and pulled the 1e6 junk into the fit.
    baseline = depth_loss(rendered, measured)
    assert torch.allclose(depth_loss(neg_bg, measured), baseline)


def test_minmax_loss_fits_affine_forward():
    rendered = _raster_ramp(_masked_depth({}))
    measured = 3.0 * rendered + 2.0
    loss = depth_loss(rendered, measured, mode="minmax")
    assert loss.item() < 1e-4


def test_minmax_vs_affine_inverted():
    rendered = _raster_ramp(_masked_depth({}))
    measured = 1.0 - rendered
    affine = depth_loss(rendered, measured, mode="affine")
    minmax = depth_loss(rendered, measured, mode="minmax")
    assert affine.item() < 1e-5
    assert minmax.item() > 0.01


def test_loss_fully_background_returns_zero():
    rendered = torch.full((2, 1, 8, 8), NO_DEPTH)
    measured = torch.rand(2, 1, 8, 8)
    loss = depth_loss(rendered, measured)
    assert loss.item() == 0.0
    assert depth_loss(rendered, measured, reduction="none").numel() == 0


def test_loss_reduction_none_skips_empty_images():
    rendered = _raster_ramp(_masked_depth({}))
    rendered_full = torch.cat(
        [rendered, torch.full_like(rendered, NO_DEPTH)], dim=0
    )  # second image fully background
    measured = 2.5 * rendered + 0.3
    measured_full = torch.cat([measured, torch.rand_like(measured)], dim=0)
    per_image = depth_loss(rendered_full, measured_full, reduction="none")
    assert per_image.shape == (1,)  # only the image with geometry
    single = depth_loss(rendered, measured, reduction="none")
    assert torch.allclose(per_image, single)


def test_loss_invalid_mode():
    rendered = _raster_ramp(_masked_depth({}))
    with pytest.raises(ValueError, match="invalid"):
        depth_loss(rendered, rendered, mode="bogus")


def test_loss_gradients_flow():
    rendered = _raster_ramp(_masked_depth({}))
    measured = (2.0 * rendered + 0.1).clone().requires_grad_(True)
    loss = depth_loss(rendered, measured)
    loss.backward()
    assert measured.grad is not None
    assert torch.isfinite(measured.grad).all()
    # Only valid (rendered) pixels carry gradients.
    assert measured.grad[0, 0][rendered[0, 0] == NO_DEPTH].abs().sum() == 0


# ------------------------------------------------ real-render sentinel test


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Renderer requires a CUDA device"
)
def test_render_depth_background_sentinel():
    from vdsds.rendering import Camera, LightSource, Renderer
    from vdsds.representations.mesh import Mesh

    mesh = Mesh(
        V=torch.tensor(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]
        ),
        F=torch.tensor([[0, 1, 2]], dtype=torch.int32),
    ).to("cuda:0")
    renderer = Renderer.depth(torch.device("cuda:0"))
    # The training/viewer convention renders origin-centered meshes with
    # random orbit cameras (the default identity camera faces away).
    camera, depth = None, None
    for _ in range(5):  # a pose may be edge-on; reject those seeds
        camera = Camera.random_rot(radius=1.5, point_upwards=True).to("cuda:0")
        out = renderer.render(mesh, camera, LightSource())
        depth = out["depth"]
        if (depth[0, 0] != NO_DEPTH).any():
            break
    assert depth.shape == (1, 1, camera.H, camera.W)
    # Background pixels hold the sentinel exactly; foreground is valid.
    fg = depth[0, 0][depth[0, 0] != NO_DEPTH]
    assert fg.numel() > 0
    assert ((fg > 0) & (fg <= 1)).all()


# ------------------------------------------- constraint-image depth queries


def test_depth_from_constraints_uses_placed_overlays(fake_model, tmp_path):
    from vdsds.constraints import Constraint
    from vdsds.rendering.camera import Camera
    from vdsds.rendering.view_state import Overlay, ViewState

    # A 100x100 opaque png placed into a 64x48 viewport.
    png = tmp_path / "a.png"
    arr = np.zeros((100, 100, 4), np.uint8)
    arr[..., 0] = 255
    arr[..., 3] = 255
    Image.fromarray(arr, mode="RGBA").save(png)

    camera = Camera(H=64, W=48)
    constraints = {
        # image-bearing: queried through overlay.image() placement
        "img": Constraint(
            view=ViewState(camera=camera, overlay=Overlay(str(png))), prompt=""
        ),
        # no image: not queried
        "view_only": Constraint(view=ViewState(camera=camera.copy()), prompt="y"),
    }
    depths = DepthAnything.depth_from_constraints(constraints)
    assert set(depths) == {"img"}
    d = depths["img"]
    assert d.shape == (60, 80)  # fake model's fixed output size
    assert d.dtype == torch.float32
    # The DA v3 input is the PLACED overlay (viewport-sized), not the
    # raw stored png.
    pil = fake_model.inputs[0][0]
    assert pil.size == (48, 64)


def test_depth_from_constraints_empty_when_no_images(fake_model):
    from vdsds.constraints import Constraint
    from vdsds.rendering.camera import Camera
    from vdsds.rendering.view_state import ViewState

    constraints = {
        "view_only": Constraint(view=ViewState(camera=Camera()), prompt="y")
    }
    assert DepthAnything.depth_from_constraints(constraints) == {}
    assert fake_model.loads == []  # DA v3 never loaded without images


# --------------------------------------------------------- lazy importing


def test_load_is_lazily_imported(monkeypatch):
    has_package = importlib.util.find_spec("depth_anything_3") is not None
    if has_package:
        pytest.skip("depth_anything_3 is installed in this env")
    with pytest.raises(ImportError, match="depth_anything_3"):
        DepthAnything._load_model()


def test_module_imports_without_depth_anything():
    # The wrapper itself must not need the package at import time.
    import vdsds.utils.depth  # noqa: F401
