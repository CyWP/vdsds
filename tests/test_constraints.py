import os

# The View test creates widgets (and a QApplication); force the offscreen
# platform before any Qt import so the tests run headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from vdsds.constraints import ConstraintManager
from vdsds.rendering.camera import Camera, CameraCoordinates
from vdsds.rendering.view_state import (
    Overlay,
    ViewState,
    camera_from_dict,
    camera_to_dict,
)
from vdsds.utils.config import Config
from vdsds.utils.quaternion import Quaternion


# ---------------------------------------------------------------- fixtures


def _make_png(path, size=100):
    """Solid red, fully opaque image with RGBA."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 3] = 255
    Image.fromarray(arr, mode="RGBA").save(path)


@pytest.fixture
def png_path(tmp_path):
    path = tmp_path / "overlay.png"
    _make_png(path)
    return path


@pytest.fixture
def camera():
    Q = Quaternion(torch.tensor([0.7, 0.1, 0.2, 0.3]))
    co = CameraCoordinates(origin=torch.tensor([0.5, -0.25, 1.0]), Q=Q, radius=1.75)
    return Camera(H=512, W=512, F=60, Zn=0.1, Zf=100.0, co=co)


@pytest.fixture
def overlay(png_path):
    return Overlay(str(png_path), scale=1.2, x=7, y=-5, rot=30.0, alpha=0.8)


@pytest.fixture
def view_state(camera, overlay):
    return ViewState(camera=camera.copy(), overlay=overlay)


# ------------------------------------------------------------- camera dict


def test_camera_dict_roundtrip(camera):
    restored = camera_from_dict(camera_to_dict(camera))
    assert restored.H == camera.H
    assert restored.W == camera.W
    assert restored.F == camera.F
    assert restored.Zn == camera.Zn
    assert restored.Zf == camera.Zf
    assert torch.allclose(restored.co.Q.q, camera.co.Q.q, atol=1e-6)
    assert torch.equal(restored.co.origin, camera.co.origin)
    assert restored.co.radius == camera.co.radius


def test_camera_dict_json_types(camera):
    data = camera_to_dict(camera)
    text = json.dumps(data)  # must be JSON-serializable
    assert isinstance(json.loads(text)["co"]["Q"], list)


# ------------------------------------------------------------ viewstate


def test_viewstate_roundtrip(view_state):
    restored = ViewState.from_dict(view_state.to_dict())
    assert torch.allclose(restored.camera.co.Q.q, view_state.camera.co.Q.q, atol=1e-6)
    assert torch.equal(restored.camera.co.origin, view_state.camera.co.origin)
    ov, res_ov = view_state.overlay, restored.overlay
    assert res_ov is not None
    assert (res_ov.scale, res_ov.x, res_ov.y, res_ov.rot, res_ov.alpha) == (
        ov.scale,
        ov.x,
        ov.y,
        ov.rot,
        ov.alpha,
    )


def test_viewstate_without_overlay_roundtrip(camera):
    state = ViewState(camera=camera)
    data = state.to_dict()
    assert data["overlay"] is None
    restored = ViewState.from_dict(data)
    assert restored.overlay is None


# --------------------------------------------------------------- overlay


def test_overlay_roundtrip(png_path, overlay):
    data = overlay.to_dict()
    restored = Overlay.from_dict(data)
    assert restored.img.image().shape == overlay.img.image().shape
    assert torch.equal(restored.img.image(), overlay.img.image())


def test_overlay_manipulators(png_path):
    ov = Overlay(str(png_path))
    assert (ov.x, ov.y, ov.rot, ov.scale) == (0, 0, 0.0, 1.0)
    ov.translate(12, -3)
    assert (ov.x, ov.y) == (12, -3)
    ov.rotate(15.0)
    assert ov.rot == 15.0
    ov.scale_by(math.exp(-120 * 1e-4))
    assert ov.scale == pytest.approx(math.exp(-120 * 1e-4))


def test_overlay_image_identity_fills(png_path):
    ov = Overlay(str(png_path))  # 100x100, defaults
    canvas = ov.image(64, 64)
    assert canvas.shape == (1, 4, 64, 64)
    # The image is padded to its diagonal square before placement, so the
    # corners of the canvas are (transparent) padding; the center is the
    # opaque red image.
    assert canvas[0, 3, 0, 0].item() == pytest.approx(0.0)
    center = canvas[0, :, 32, 32]
    assert center[3].item() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(center[:3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


def test_overlay_image_translate_clips(png_path):
    ov = Overlay(str(png_path), x=999, y=999)
    canvas = ov.image(64, 64)
    assert torch.equal(canvas, torch.zeros_like(canvas))


def test_overlay_composite(png_path):
    ov = Overlay(str(png_path), alpha=0.35)
    bg = torch.full((1, 4, 64, 64), 0.5)
    render = bg.clone()
    out = ov.overlay(render)
    assert out.shape == (1, 4, 64, 64)
    # Center: opaque overlay blended with alpha 0.35:
    # 0.35 * (1, 0, 0) + 0.65 * (0.5, 0.5, 0.5).
    center = out[0, :, 32, 32]
    assert torch.allclose(center[:3], torch.tensor([0.675, 0.325, 0.325]), atol=1e-4)
    # Corner: transparent overlay padding keeps the background.
    corner = out[0, :, 0, 0]
    assert torch.allclose(corner[:3], bg[0, :3, 0, 0])


def test_overlay_copy_is_independent(png_path, overlay):
    clone = overlay.copy()
    clone.translate(5, 5)
    assert overlay.x == 7
    assert clone.x == 12
    # The Splimage is a deep copy: mutating the clone's tensor must not
    # touch the original.
    clone.img.image()[0, 0, 0, 0] = 0.5
    assert overlay.img.image()[0, 0, 0, 0].item() == 1.0


def test_overlay_from_tensor_cannot_serialize():
    with pytest.raises(ValueError):
        Overlay(torch.zeros(1, 4, 8, 8)).to_dict()


# ---------------------------------------------------------------- manager


def test_manager_roundtrip(tmp_path, camera, overlay):
    path = tmp_path / "constraints.json"
    manager = ConstraintManager(path)
    cid1 = manager.add(
        ViewState(camera=camera.copy(), overlay=overlay), prompt="a big jersey cow"
    )
    cid2 = manager.add(ViewState(camera=Camera()), prompt="")
    assert len(manager) == 2

    fresh = ConstraintManager(path)
    assert set(fresh.items()) == {cid1, cid2}
    c = fresh.get(cid1)
    assert c is not None
    assert c.prompt == "a big jersey cow"
    assert torch.allclose(c.view.camera.co.Q.q, camera.co.Q.q, atol=1e-6)
    assert torch.equal(c.view.camera.co.origin, camera.co.origin)
    ov = c.view.overlay
    assert (ov.scale, ov.x, ov.y, ov.rot, ov.alpha) == (1.2, 7, -5, 30.0, 0.8)
    # The unnamed constraint has no overlay.
    assert fresh.get(cid2).view.overlay is None


def test_manager_missing_file_is_empty(tmp_path):
    manager = ConstraintManager(tmp_path / "missing.json")
    assert len(manager) == 0
    # Saving via add creates the file.
    manager.add(ViewState(camera=Camera()))
    assert (tmp_path / "missing.json").exists()


def test_manager_corrupt_file_is_empty(tmp_path):
    path = tmp_path / "constraints.json"
    path.write_text("{not json")
    manager = ConstraintManager(path)
    assert len(manager) == 0


def test_manager_remove_and_update(tmp_path, view_state):
    path = tmp_path / "constraints.json"
    manager = ConstraintManager(path)
    cid1 = manager.add(view_state.copy(), prompt="first")
    cid2 = manager.add(ViewState(camera=Camera()), prompt="second")

    manager.remove(cid1)
    assert cid1 not in manager
    assert cid2 in manager
    fresh = ConstraintManager(path)
    assert set(fresh.items()) == {cid2}

    camera = fresh.get(cid2).view.camera
    camera.co.origin += 0.25
    fresh.set_view(cid2, ViewState(camera=camera))
    fresh.set_prompt(cid2, "rewritten")
    again = ConstraintManager(path)
    assert again.get(cid2).prompt == "rewritten"
    assert torch.allclose(
        again.get(cid2).view.camera.co.origin, camera.co.origin, atol=1e-6
    )


def test_manager_no_path_is_in_memory(view_state):
    manager = ConstraintManager()
    cid = manager.add(view_state.copy(), prompt="transient")
    assert cid in manager
    assert manager.path is None


def test_manager_save_exports_without_switching_path(tmp_path, view_state):
    path = tmp_path / "constraints.json"
    manager = ConstraintManager(path)
    cid1 = manager.add(view_state.copy(), prompt="first")
    cid2 = manager.add(ViewState(camera=Camera()), prompt="second")
    export = tmp_path / "export.json"

    manager.save(export)
    assert json.loads(export.read_text()).keys() == {cid1, cid2}

    # Autosave target stays the original file: further mutations land
    # there but not in the export.
    cid3 = manager.add(ViewState(camera=Camera()), prompt="third")
    assert json.loads(path.read_text()).keys() == {cid1, cid2, cid3}
    assert json.loads(export.read_text()).keys() == {cid1, cid2}


# --------------------------------------------------- CLI / script defaults


def test_cli_constraints_flag_maps_to_path():
    from vdsds.__main__ import parse_args

    args, config = parse_args(["--task", "view_model", "--constraints", "c.json"])
    assert config["path"]["constraints"] == "c.json"


def test_cli_without_constraints_flag_injects_nothing():
    from vdsds.__main__ import parse_args

    args, config = parse_args(["--task", "view_model"])
    # Nothing is injected; the script default decides (None => in-memory).
    assert "path" not in config


def test_default_constraints_path_is_none():
    from vdsds.scripts.base import ViewableScript

    assert ViewableScript._default_config_overrides["path"]["constraints"] is None


# ------------------------------------------------- train script loading


def test_train_script_loads_constraints_from_json_only(tmp_path):
    """Constraints are loaded from the JSON file and nothing else: a
    populated view manager must not leak into the loader."""
    from types import SimpleNamespace

    from vdsds.scripts.train_sds import TrainModelSDS

    path = tmp_path / "constraints.json"
    cid = ConstraintManager(path).add(ViewState(camera=Camera()), prompt="train file")
    script = TrainModelSDS.__new__(TrainModelSDS)
    script.view = SimpleNamespace(
        constraints=ConstraintManager(tmp_path / "other.json")  # non-empty, unused
    )
    script.view.constraints.add(ViewState(camera=Camera()), prompt="session")
    script.config = Config({"path": {"constraints": str(path)}})
    constraints = script.get_constraints()
    assert set(constraints) == {cid}
    assert constraints[cid].prompt == "train file"


def test_train_script_without_constraints_is_empty():
    from vdsds.scripts.train_sds import TrainModelSDS
    from vdsds.utils.config import Config

    script = TrainModelSDS.__new__(TrainModelSDS)
    script.config = Config({})
    assert script.get_constraints() == {}


# ------------------------------------ constraint-driven prompts (train script)


class _FakeDF:
    def encode_text_2(self, prompts, negative_prompt=None, batch_size=1):
        return torch.zeros(len(prompts) * batch_size, 1)


def _train_script(constraints, centroid=None, device="cuda:0"):
    """Bare TrainModelSDS instance (skips the model-loading __init__)."""
    from types import SimpleNamespace

    from vdsds.scripts.train_sds import TrainModelSDS

    device = torch.device(device)
    script = TrainModelSDS.__new__(TrainModelSDS)
    script.constraints = constraints
    script.model = SimpleNamespace(
        device=device,  # TrainModelSDS.device delegates here
        model=SimpleNamespace(
            centroid=(
                torch.zeros(3, device=device) if centroid is None else centroid
            )
        ),
    )
    return script


def _build_constraints(pairs):
    """pairs: list of (prompt, offset) -> constraints with identity-Q
    cameras of radius 1, translated to the given offset."""
    from vdsds.constraints import Constraint
    from vdsds.rendering.camera import Camera, CameraCoordinates
    from vdsds.utils.quaternion import Quaternion

    constraints = {}
    for i, (prompt, offset) in enumerate(pairs):
        camera = Camera(
            co=CameraCoordinates(
                origin=torch.zeros(3),
                Q=Quaternion(torch.tensor([1.0, 0, 0, 0])),
                radius=1.0,
            )
        )
        camera.translate(offset)  # location = offset + FORWARD * radius
        constraints[f"cid_{i}"] = Constraint(
            view=ViewState(camera=camera), prompt=prompt
        )
    return constraints


def test_anchors_and_prompts_from_constraints():
    """Module-level derivation: prompts follow prompt texts (prompt-less
    constraints are excluded); anchors follow the views -- every
    constraint carries a view, so anchors exist for all of them."""
    from vdsds.constraints import (
        anchors_from_constraints,
        prompts_from_constraints,
        Constraint,
    )

    constraints = _build_constraints([("a cow", torch.zeros(3))])
    constraints["cid_v"] = Constraint(view=ViewState(camera=Camera()), prompt="")
    constraints["cid_p"] = Constraint(
        view=ViewState(camera=Camera()), prompt="text cow"
    )

    prompt_map = prompts_from_constraints(constraints)
    assert prompt_map == {"cid_0": "a cow", "cid_p": "text cow"}

    anchor_map = anchors_from_constraints(constraints)
    # Identity-Q camera at radius 1: location = (0, 0, -1).
    assert set(anchor_map) == set(constraints)
    assert torch.allclose(
        anchor_map["cid_0"], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6
    )
    # Explicit centroid shifts the direction accordingly.
    shifted = anchors_from_constraints(
        constraints, centroid=torch.tensor([0.0, 1.0, 0.0])
    )
    assert torch.allclose(
        shifted["cid_0"],
        torch.tensor([0.0, 1.0, 1.0]) / math.sqrt(2),
        atol=1e-6,
    )


def test_constraint_prompts_and_anchor_directions():
    """Prompts and anchors are aligned one-to-one over the
    prompt-bearing constraints (all of which carry views)."""
    offsets = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 2.0, 0.0])]
    constraints = _build_constraints([("left cow", offsets[0]), ("", offsets[1])])
    script = _train_script(constraints, device="cpu")

    prompts, anchors = script.get_constraint_prompts()
    # The prompt-less constraint is not taken into account.
    assert prompts == ["left cow"]
    # Anchor = normalized (centroid - camera.location); aligned.
    camera = constraints["cid_0"].view.camera.copy().to(script.device)
    expected = (script.model.model.centroid - camera.location)
    expected = (expected / expected.norm()).tolist()
    assert torch.allclose(
        torch.tensor(anchors[0]), torch.tensor(expected), atol=1e-6
    )
    assert len(anchors) == len(prompts) == 1


def test_get_text_embeds_uses_constraint_prompts():
    offsets = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
    constraints = _build_constraints([("a moose", offsets[0]), ("cow", offsets[1])])
    script = _train_script(constraints, device="cpu")
    script.config = Config(
        {
            "prompt": {
                "text": ["config prompt"],
                "anchors": [[1.0, 0.0, 0.0]],
                "extra": [],
                "negative": None,
            },
            "camera": {"views": 1, "aug_views": 0},
            "diffusion": {"loss": "sds"},
        }
    )
    prompts, embeds, basis = script.get_text_embeds(_FakeDF())
    assert basis is None  # 'sds' loss: no BSD basis
    assert prompts[0].startswith("a moose")
    assert prompts[1] == "cow"  # last prompt kept raw (MeshUp convention)
    assert all("config prompt" not in p for p in prompts)
    assert embeds.shape[0] == len(prompts)


def test_get_text_embeds_bsd_pairs_prompts_with_anchors():
    constraints = _build_constraints([("a moose", torch.zeros(3))])
    script = _train_script(constraints, device="cpu")
    script.config = Config(
        {
            "prompt": {
                "text": ["config prompt"],
                "anchors": [[1.0, 0.0, 0.0]],
                "extra": [],
                "negative": None,
                "overlap": 0.75,
                "base": None,
            },
            "camera": {"views": 1, "aug_views": 0},
            "diffusion": {"loss": "bsd"},
        }
    )
    prompts, embeds, basis = script.get_text_embeds(_FakeDF())
    # BSD concept prompt from the constraint (prompt+view are aligned);
    # the trailing "" branch is appended as in the config path.
    assert prompts == ["a moose", ""]
    assert basis is not None
    # Anchor direction: centroid - location = (0, 0, 1) for the
    # identity-Q camera at radius 1.
    assert torch.allclose(
        basis.centroids.flatten(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6
    )
    assert embeds.shape[0] == len(prompts)


def test_regularizer_losses_wire_constraint_steps():
    script = _train_script({}, device="cpu")
    script.config = Config(
        {
            "loss": {
                "diffusion": 1.0,
                "jacobian": 0.0,
                "composed_jacobian": 0.0,
                "view_dependence": 0.0,
                "laplacian": 0.0,
                "volume": 0.0,
                "volume_eval_points": 8,
                "silhouette": 2.0,
                "depth": 0.5,
            }
        }
    )
    script.silhouette_loss = lambda: torch.tensor(3.0)
    script.depth_loss = lambda: torch.tensor(1.0)
    regs = script._regularizer_losses(cameras=[], accum_steps=2)
    assert set(regs) == {"silhouette", "depth"}
    assert regs["silhouette"][1] == 2.0 and regs["depth"][1] == 0.5
    assert regs["silhouette"][0].item() == 3.0
    assert regs["depth"][0].item() == 1.0


def test_regularizer_losses_skip_zero_constraint_steps():
    script = _train_script({}, device="cpu")
    script.config = Config(
        {
            "loss": {
                "diffusion": 1.0,
                "jacobian": 0.0,
                "composed_jacobian": 0.0,
                "view_dependence": 0.0,
                "laplacian": 0.0,
                "volume": 0.0,
                "volume_eval_points": 8,
                "silhouette": 0.0,
                "depth": 0.0,
            }
        }
    )
    regs = script._regularizer_losses(cameras=[], accum_steps=2)
    assert regs == {}


# ------------------------------------------- view interactions (headless)


@pytest.fixture
def no_frame_loop(monkeypatch):
    """Keep the View's background frame retriever from firing (no model to
    render in the interaction test)."""
    from vdsds.utils.timer import TimedJob

    monkeypatch.setattr(TimedJob, "start", lambda self, action=None, frequency=None: None)


class _FakeHost:
    """Minimal stand-in for a Model: enough for ObjViewer's camera."""

    device = torch.device("cpu")
    dtype = torch.float32


def test_view_interactions(tmp_path, png_path, no_frame_loop):
    from vdsds.view.keymap import K_ALT, K_CTRL
    from vdsds.view.view import View

    constraints_path = tmp_path / "view_constraints.json"
    view = View(_FakeHost(), fps=1, constraints_path=constraints_path)
    view.frame_retriever.stop()  # no unused model to render in this test
    try:
        assert len(view.constraints) == 0
        assert view.window.window.constraint_list.count() == 0
        action_btns = (
            view.window.window.constraint_set_view_btn,
            view.window.window.constraint_jump_btn,
            view.window.window.constraint_prompt_btn,
            view.window.window.constraint_delete_btn,
        )
        # No selection: per-constraint actions are disabled.
        assert all(not btn.isEnabled() for btn in action_btns)

        # ----- overlay state and live manipulation hooks -----
        view.load_overlay(png_path)
        ov = view.view_state.overlay
        assert ov is not None

        view._mouse_drag(12, 7, [K_ALT])
        assert (ov.x, ov.y) == (12, 7)
        # With Alt, the camera untouched (drag counters cleared on action).
        assert view.viewer.rot_x == 0 and view.viewer.rot_y == 0

        view._mouse_drag(3, 5, [K_ALT, K_CTRL])
        assert ov.rot == 3

        view._mouse_scroll(-160, [K_ALT])
        assert ov.scale == pytest.approx(math.exp(-160 * 1e-3))

        # Unmodified keys fall through to the viewer as before.
        view._mouse_drag(5, 2, [])
        assert view.viewer.rot_x == 5 and view.viewer.rot_y == 2
        assert (ov.x, ov.y) == (12, 7)  # untouched without Alt

        # ----- add constraint captures camera + overlay state -----
        origin0 = view.viewer.camera.co.origin.clone()
        cid = view.add_constraint("a moose")
        assert cid in view.constraints
        assert view.window.window.constraint_list.count() == 1
        rec = view.constraints.get(cid)
        assert rec.prompt == "a moose"
        assert torch.equal(rec.view.camera.co.origin, origin0)
        assert rec.view.overlay.x == 12
        assert rec.view.overlay.rot == 3
        # Saved immediately:
        data = json.loads(constraints_path.read_text())
        assert set(data) == {cid}
        assert data[cid]["prompt"] == "a moose"
        assert data[cid]["view"]["overlay"]["x"] == 12

        # ----- set view resets the constraint to the current view -----
        view.viewer.camera.translate(torch.tensor([0.5, 0.0, 0.0]))
        view.set_constraint_view(cid)
        rec2 = view.constraints.get(cid)
        assert torch.allclose(rec2.view.camera.co.origin, view.viewer.camera.co.origin)
        data = json.loads(constraints_path.read_text())
        assert data[cid]["view"]["camera"]["co"]["origin"] == pytest.approx(
            [v for v in (origin0 + torch.tensor([0.5, 0.0, 0.0])).tolist()]
        )

        # ----- set prompt -----
        view.set_constraint_prompt(cid, "updated")
        assert view.constraints.get(cid).prompt == "updated"
        item = view.window.window.constraint_list.item(0)
        assert item.text().startswith("updated")
        # The entry line shows the overlay image name, toolTip the path.
        assert "overlay.png" in item.text()
        assert item.toolTip().endswith("overlay.png")

        # ----- jump to view restores camera + overlay -----
        view.viewer.camera.translate(torch.tensor([-0.25, 0.0, 0.0]))
        view.jump_constraint(cid)
        assert torch.equal(view.viewer.camera.co.origin, rec2.view.camera.co.origin)
        assert view.view_state.camera is view.viewer.camera
        assert view.view_state.overlay.x == 12
        assert view.view_state.overlay.rot == 3

        # ----- save JSON (export; autosaves keep their own file) -----
        export_path = tmp_path / "export_copy.json"
        view.save_constraints(export_path)
        exported = json.loads(export_path.read_text())
        assert set(exported) == {cid}
        assert exported[cid]["prompt"] == "updated"

        # ----- delete -----
        view.delete_constraint(cid)
        assert len(view.constraints) == 0
        assert view.window.window.constraint_list.count() == 0
        data = json.loads(constraints_path.read_text())
        assert data == {}
        assert all(not btn.isEnabled() for btn in action_btns)

        # ----- real Qt wheel path: vertical zooms (Alt), tilt is inert -----
        from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
        from PySide6.QtGui import QKeyEvent, QWheelEvent

        w = view.window.window
        vp = w.viewport
        w.keyPressEvent(QKeyEvent(QEvent.KeyPress, K_ALT, Qt.NoModifier))
        radius0 = view.viewer.camera.co.radius

        def _wheel(dx: int, dy: int) -> QWheelEvent:
            return QWheelEvent(
                QPointF(5.0, 5.0),
                QPointF(5.0, 5.0),
                QPoint(0, 0),
                QPoint(dx, dy),
                Qt.NoButton,
                Qt.AltModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )

        scale0 = view.view_state.overlay.scale
        vp.wheelEvent(_wheel(0, 120))
        assert view.view_state.overlay.scale != scale0  # vertical wheel zooms
        assert view.viewer.camera.co.radius == radius0  # Alt keeps camera out

        # Horizontal-only (wheel tilt): no zoom, no camera translation.
        scale_before_tilt = view.view_state.overlay.scale
        vp.wheelEvent(_wheel(120, 0))
        assert view.view_state.overlay.scale == scale_before_tilt
        assert view.viewer.camera.co.radius == radius0

        w.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, K_ALT, Qt.NoModifier))
    finally:
        view.close()


# --------------------------------------- depth-loss method (train script)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Renderer requires a CUDA device"
)
def test_train_script_depth_loss_uses_constraint_images(monkeypatch, tmp_path):
    import numpy as np

    import vdsds.utils.depth as depth_module
    from vdsds.constraints import Constraint
    from vdsds.rendering import Camera, LightSource
    from vdsds.rendering.view_state import Overlay
    from vdsds.representations.mesh import Mesh

    def _png(path, alpha_area=20):
        arr = np.zeros((100, 100, 4), np.uint8)
        arr[alpha_area : 100 - alpha_area, alpha_area : 100 - alpha_area, :3] = 255
        arr[alpha_area : 100 - alpha_area, alpha_area : 100 - alpha_area, 3] = 255
        Image.fromarray(arr, mode="RGBA").save(path)

    png_a = tmp_path / "sketch_a.png"
    _png(png_a)

    mesh = Mesh(
        V=torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]),
        F=torch.tensor([[0, 1, 2]], dtype=torch.int32),
    ).to("cuda:0")
    camera = Camera.random_rot(radius=1.5, point_upwards=True).to("cuda:0")
    constraints = {
        # image + view: full depth step participant
        "cid_d": Constraint(
            view=ViewState(camera=camera, overlay=Overlay(str(png_a))), prompt=""
        ),
        # view only (no image): no DA v3 depth, not matched
        "cid_view_only": Constraint(
            view=ViewState(camera=camera.copy()), prompt=""
        ),
    }
    script = _train_script(constraints)
    script.config = Config(
        {"lighting": {"alignment": "camera", "jitter_sigma": 0.0, "strength": 1.0}}
    )
    script.model.deformed = lambda cam: mesh
    script.randomized_light = lambda cam: LightSource().to(script.model.device)

    da_calls = []
    loss_calls = []

    def fake_from_constraints(cons):
        da_calls.append(1)
        return {"cid_d": torch.rand(60, 80)}

    def fake_depth_loss(rendered, measured, mode="affine", **kwargs):
        loss_calls.append((rendered, measured, mode))
        return torch.tensor(0.1234, device=rendered.device)

    monkeypatch.setattr(
        depth_module.DepthAnything,
        "depth_from_constraints",
        classmethod(lambda cls, cons, **kw: fake_from_constraints(cons)),
    )
    monkeypatch.setattr(depth_module, "depth_loss", fake_depth_loss)

    # Precompute before the run (what run() does when loss.depth is on):
    # the DA v3 load/infer/unload cycle happens inside this call.
    script.prepare_constraint_depths()
    assert len(da_calls) == 1

    loss = script.depth_loss()
    assert loss.item() == pytest.approx(0.1234)
    rendered, measured, mode = loss_calls[0]
    assert rendered.shape == (1, 1, camera.H, camera.W)
    assert measured.shape == (1, 1, camera.H, camera.W)
    assert rendered.device.type == "cuda"
    assert torch.isfinite(rendered).all()
    assert mode == "affine"
    # No further DA v3 queries: the maps were precomputed before the run.
    script.depth_loss()
    assert len(da_calls) == 1
    assert len(loss_calls) == 2


def test_train_script_depth_loss_requires_precomputed_depths():
    script = _train_script(
        _build_constraints([("a cow", torch.zeros(3))]), device="cpu"
    )
    with pytest.raises(ValueError, match="not precomputed"):
        script.depth_loss()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Renderer requires a CUDA device"
)
def test_train_script_silhouette_loss_over_constraint_overlays(monkeypatch, tmp_path):
    import numpy as np

    import vdsds.utils.silhouette as silhouette_module
    from vdsds.constraints import Constraint
    from vdsds.rendering import Camera
    from vdsds.rendering.view_state import Overlay, camera_to_dict
    from vdsds.representations.mesh import Mesh

    # Overlay png: opaque center (alpha 1) with transparent border.
    png = tmp_path / "silhouette.png"
    arr = np.zeros((100, 100, 4), np.uint8)
    arr[20:80, 20:80, :3] = 255
    arr[20:80, 20:80, 3] = 255
    Image.fromarray(arr, mode="RGBA").save(png)

    mesh = Mesh(
        V=torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]),
        F=torch.tensor([[0, 1, 2]], dtype=torch.int32),
    ).to("cuda:0")
    camera = Camera.random_rot(radius=1.5, point_upwards=True).to("cuda:0")
    constraints = {
        "cid_s": Constraint(
            view=ViewState(camera=camera, overlay=Overlay(str(png))),
            prompt="",
        ),
        # An overlay-less constraint must be skipped.
        "cid_no_overlay": Constraint(view=ViewState(camera=camera.copy()), prompt=""),
    }
    script = _train_script(constraints)
    script.config = Config({})
    script.model.deformed = lambda cam: mesh

    calls = []

    def fake_silhouette(tgt_imgs, meshes, cameras, **kwargs):
        calls.append((tgt_imgs, meshes, cameras, kwargs))
        return torch.tensor(0.6789, device=tgt_imgs.device)

    monkeypatch.setattr(silhouette_module, "silhouette_loss", fake_silhouette)

    loss = script.silhouette_loss()
    assert loss.item() == pytest.approx(0.6789)
    # Single qualified constraint: batch of one RGBA target image.
    tgt_imgs, meshes_in, cams, extra = calls[0]
    assert extra == {"in_weight": 0.0}  # config default: inverse term off
    assert tgt_imgs.shape == (1, 4, camera.H, camera.W)
    assert tgt_imgs.device.type == "cuda"
    mask = tgt_imgs[0, 3]
    assert (mask > 0).any() and (mask == 0).any()
    # The deformed mesh and constraint camera are forwarded per image,
    # pairing with the target.
    assert meshes_in == [mesh]
    assert camera_to_dict(cams[0]) == camera_to_dict(camera)


def test_train_script_silhouette_inverse_weight_from_config(monkeypatch, tmp_path):
    import numpy as np

    import vdsds.utils.silhouette as silhouette_module
    from vdsds.constraints import Constraint
    from vdsds.rendering import Camera
    from vdsds.rendering.view_state import Overlay

    png = tmp_path / "silhouette.png"
    arr = np.zeros((32, 32, 4), np.uint8)
    arr[..., :3] = 255
    arr[..., 3] = 255
    Image.fromarray(arr, mode="RGBA").save(png)

    camera = Camera(H=32, W=32)
    constraints = {
        "cid_s": Constraint(
            view=ViewState(camera=camera, overlay=Overlay(str(png))), prompt=""
        )
    }
    script = _train_script(constraints, device="cpu")
    script.config = Config({"loss": {"silhouette_inverse": 0.4}})
    script.model.deformed = lambda cam: Camera.random_rot(radius=1.5)

    calls = []

    def fake_silhouette(tgt_imgs, meshes, cameras, **kwargs):
        calls.append(kwargs)
        return torch.tensor(0.5)

    monkeypatch.setattr(silhouette_module, "silhouette_loss", fake_silhouette)
    script.silhouette_loss()
    assert calls[0] == {"in_weight": 0.4}


def test_train_script_silhouette_loss_requires_overlays():
    from vdsds.constraints import Constraint

    script = _train_script(
        {"cid_x": Constraint(view=ViewState(camera=Camera()), prompt="")}
    )
    with pytest.raises(ValueError, match="requires constraints with overlays"):
        script.silhouette_loss()
