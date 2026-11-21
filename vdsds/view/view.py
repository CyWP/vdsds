import math
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog

from ..representations.constraints import ConstraintManager
from ..deformations.base import Deformation
from ..rendering.camera import Camera
from ..rendering.light import LightSource
from ..rendering.view_state import Overlay, ViewState
from ..representations.base import Model
from ..utils.img import ImgUtils
from ..utils.timer import TimedJob
from .keymap import K_ALT, K_CTRL, KeyHandler
from .obj_view import ObjViewer
from .window import AppView

_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp)"


def draw_text_overlay(image: np.ndarray, text: str) -> np.ndarray:
    """Draws a top-left text overlay onto an HxWx4 uint8 frame.

    Black text with a white offset shadow, so it stays readable on any
    background. Returns the modified array (in place).
    """
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for dx, dy, color in ((1, 1, (255, 255, 255)), (0, 0, (0, 0, 0))):
        draw.multiline_text((8 + dx, 8 + dy), text, fill=color)
    return np.asarray(pil)


class View(QApplication):
    frame_ready = Signal(np.ndarray)

    def __init__(
        self,
        obj: Model | Deformation,
        light: LightSource | None = None,
        camera: Camera | None = None,
        fps: int = 24,
        on_close: Callable | None = None,
        constraints_path: Path | str | None = None,
        overlay: Callable[[Camera], str | None] | None = None,
    ):
        """Live viewer for a model or deformation.

        Args:
            obj: Model or deformation to display.
            light: Optional fixed light source.
            camera: Optional initial camera.
            fps: Frame rate of the render/update timer.
            on_close: Callback invoked when the window is closed.
            overlay: Optional per-frame text hook, called with the
                viewer's current camera; the returned string is drawn on
                the frame (multi-line supported). Returning None or ''
                draws nothing.
        """
        super().__init__()
        self.window = AppView()
        self.viewer = ObjViewer(obj, camera=camera, light=light)
        self.keys = KeyHandler()
        self.on_close = on_close
        self.overlay = overlay
        self.constraints = ConstraintManager(constraints_path)
        # The view state shares the viewer's camera reference, so orbit,
        # pan and zoom act on one camera instance only.
        self.view_state = ViewState(camera=self.viewer.camera)
        self._cid_by_row: dict[int, str] = {}
        # One-shot screenshot hook: set by the UI to catch the next
        # update() frame and save it.
        self._pending_screenshot: Path | None = None
        self._connect()
        self.frame_retriever = TimedJob()
        self.frame_retriever.start(self.update, fps)

    def _connect(self):
        w = self.window.window
        w.key_pressed.connect(self.keys.pressed)
        w.key_released.connect(self.keys.released)
        v = w.viewport
        v.drag_direction.connect(
            lambda x, y: self._mouse_drag(x, y, list(self.keys))
        )
        v.scrolled.connect(lambda x: self._mouse_scroll(x, list(self.keys)))
        v.left_clicked.connect(
            lambda x, y: self.viewer.left_click(x, y, list(self.keys))
        )
        v.right_clicked.connect(
            lambda x, y: self.viewer.right_click(x, y, list(self.keys))
        )
        if self.on_close is not None:
            w.closed.connect(self.on_close)
        self.frame_ready.connect(self.window.update)
        self.window.window.view_deformation_btn.toggled.connect(
            lambda checked: setattr(self.viewer, "view_deformed", checked)
        )
        self.window.window.record_btn.toggled.connect(self._toggle_recording)
        self.window.window.screenshot_btn.clicked.connect(self._screenshot_dialog)
        self._connect_constraints()

    # ----- Overlay + viewport interactions -----

    def _mouse_drag(self, x: int, y: int, keys: list[int]):
        """Dispatch a viewport drag to the overlay (Alt family) or the camera.

        Alt + drag translates the overlay; Ctrl + Alt + drag rotates it.
        Any other combination keeps the existing camera behavior.
        """
        overlay = self.view_state.overlay
        if overlay is not None and K_ALT in keys:
            if K_CTRL in keys:
                overlay.rotate(x)
            else:
                overlay.translate(x, y)
            return
        self.viewer.mouse_drag(x, y, keys)

    def _mouse_scroll(self, x: int, keys: list[int]):
        """Dispatch a wheel event to the overlay (Alt) or the camera.

        Scroll-up zooms the overlay in, matching the camera's scroll
        behavior. The step size is ~11% per wheel notch (the signal
        carries +-120 per notch; trackpads inject many smaller deltas
        that accumulate naturally).
        """
        overlay = self.view_state.overlay
        if overlay is not None and K_ALT in keys:
            overlay.scale_by(math.exp(x * 1e-3))
            return
        self.viewer.scroll(x, keys)

    # ----- Overlay state -----

    def load_overlay_dialog(self):
        """Open a file dialog and set the current overlay from the choice."""
        path, _ = QFileDialog.getOpenFileName(
            self.window.window, "Load Overlay", "", _IMAGE_FILTER
        )
        if path:
            self.load_overlay(path)

    def load_overlay(self, path: str | Path):
        """Set the current overlay from an image path."""
        self.view_state.overlay = Overlay(path)

    def clear_overlay(self):
        """Remove the current overlay."""
        self.view_state.overlay = None

    # ----- Constraints -----

    def _connect_constraints(self):
        w = self.window.window
        w.overlay_load_btn.clicked.connect(self.load_overlay_dialog)
        w.overlay_clear_btn.clicked.connect(self.clear_overlay)
        w.constraint_add_btn.clicked.connect(self._add_constraint_dialog)
        w.constraint_set_view_btn.clicked.connect(self._set_constraint_view_selected)
        w.constraint_jump_btn.clicked.connect(self._jump_constraint_selected)
        w.constraint_prompt_btn.clicked.connect(self._set_constraint_prompt_selected)
        w.constraint_delete_btn.clicked.connect(self._delete_constraint_selected)
        w.constraint_save_btn.clicked.connect(self._save_constraints_dialog)
        w.constraint_list.itemSelectionChanged.connect(self._on_constraint_selection)
        self._refresh_constraint_list()

    def _add_constraint_dialog(self):
        prompt = self._request_prompt("Add Constraint", "Prompt:")
        if prompt is not None:
            self.add_constraint(prompt)

    def add_constraint(self, prompt: str = "") -> str:
        """Capture the current view state as a new constraint.

        Args:
            prompt: Optional text prompt stored with the constraint.

        Returns:
            out: The generated unique id of the new constraint.
        """
        cid = self.constraints.add(self.view_state.copy(), prompt)
        self._refresh_constraint_list()
        return cid

    def _set_constraint_view_selected(self):
        cid = self._selected_cid()
        if cid is not None:
            self.set_constraint_view(cid)

    def set_constraint_view(self, cid: str):
        """Reset a constraint's view to the current one."""
        self.constraints.set_view(cid, self.view_state.copy())

    def _jump_constraint_selected(self):
        cid = self._selected_cid()
        if cid is not None:
            self.jump_constraint(cid)

    def jump_constraint(self, cid: str):
        """Load a constraint's view into the viewport.

        Rebuilds the view state so its camera reference tracks the
        viewer's camera; a constraint view with an overlay restores the
        overlay with it. Constraints without a usable view are a no-op.
        """
        constraint = self.constraints.get(cid)
        if constraint is None:
            return
        view = constraint.view
        camera = view.camera if view is not None else None
        if camera is None:
            return
        self.viewer.camera = camera.copy().to(self.viewer.camera.device)
        self.view_state = ViewState(
            camera=self.viewer.camera,
            overlay=view.overlay.copy() if view.overlay is not None else None,
        )

    def _set_constraint_prompt_selected(self):
        cid = self._selected_cid()
        if cid is None:
            return
        prompt = self._request_prompt("Set Prompt", "Prompt:")
        if prompt is not None:
            self.set_constraint_prompt(cid, prompt)

    def set_constraint_prompt(self, cid: str, prompt: str):
        """Replace a constraint's text prompt."""
        self.constraints.set_prompt(cid, prompt)
        self._refresh_constraint_list()

    def _delete_constraint_selected(self):
        cid = self._selected_cid()
        if cid is not None:
            self.delete_constraint(cid)

    def delete_constraint(self, cid: str):
        """Remove a constraint."""
        self.constraints.remove(cid)
        self._refresh_constraint_list()

    def _save_constraints_dialog(self):
        """Open a save dialog and export the constraints to the choice."""
        path, _ = QFileDialog.getSaveFileName(
            self.window.window, "Save Constraints", "constraints.json", "JSON (*.json)"
        )
        if path:
            self.save_constraints(path)

    def save_constraints(self, path: str | Path):
        """Export the current constraints to a JSON file.

        The manager's own persistence path (if any) is unchanged, so
        autosaves keep going to the original file.
        """
        self.constraints.save(path)

    def _refresh_constraint_list(self):
        """Rebuild the constraint list widget from the manager."""
        w = self.window.window
        w.constraint_list.clear()
        self._cid_by_row = {}
        for row, (cid, constraint) in enumerate(self.constraints.items().items()):
            label = constraint.prompt or "<unnamed>"
            entry = f"{label}   {cid[:6]}"
            item_overlay = constraint.view.overlay if constraint.view else None
            if item_overlay is not None and item_overlay.path is not None:
                img = Path(item_overlay.path).name
                entry = f"{entry}   {img}"
            w.constraint_list.addItem(entry)
            item = w.constraint_list.item(row)
            if item_overlay is not None and item_overlay.path is not None:
                item.setToolTip(item_overlay.path)
            self._cid_by_row[row] = cid
        w.constraint_list.setCurrentRow(-1)
        self._on_constraint_selection()

    def _selected_cid(self) -> str | None:
        w = self.window.window
        row = w.constraint_list.currentRow()
        if row < 0:
            return None
        return self._cid_by_row.get(row, None)

    def _on_constraint_selection(self):
        w = self.window.window
        enabled = w.constraint_list.currentRow() >= 0
        for btn in (
            w.constraint_set_view_btn,
            w.constraint_jump_btn,
            w.constraint_prompt_btn,
            w.constraint_delete_btn,
        ):
            btn.setEnabled(enabled)

    def _request_prompt(self, title: str, label: str) -> str | None:
        """Show a text input dialog; None when cancelled or left empty."""
        text, ok = QInputDialog.getText(self.window.window, title, label)
        return text.strip() if ok and text.strip() else None

    # ----- Frame loop / lifecycle -----

    def _toggle_recording(self, checked: bool):
        if checked:
            self.viewer.start_recording()
        else:
            path = self.viewer.stop_recording()
            if path:
                print(f"Recording saved to {path}")

    # ----- Screenshot -----

    def _screenshot_dialog(self):
        """Open a save dialog and arm the screenshot hook on the choice."""
        path, _ = QFileDialog.getSaveFileName(
            self.window.window, "Save Screenshot", "screenshot.png", _IMAGE_FILTER
        )
        if path:
            self.request_screenshot(path)

    def request_screenshot(self, path: str | Path):
        """Arm a one-shot hook that saves the next rendered frame.

        Frames are produced by the background update loop, so the
        screenshot is not taken here; instead the next ``update`` call
        is caught and its image written to ``path``.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".png")
        self._pending_screenshot = path

    def _save_screenshot(self, path: Path, render):
        """Write a rendered frame tensor to an image file.

        The render is RGBA with a transparent background; for formats
        without alpha (JPEG), it is composited over white.
        """
        img = ImgUtils.tensor2pil(render)
        if isinstance(img, list):
            img = img[0]
        if img.mode == "RGBA" and path.suffix.lower() in (".jpg", ".jpeg"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        img.save(str(path))
        print(f"Screenshot saved to {path}")

    def close(self):
        self.constraints.save()
        self.frame_retriever.stop()
        self.window.window.close()

    def update(self):
        try:
            image = self.viewer.get_render(
                *self.window.dimensions, overlay=self.view_state.overlay
            )
            if self._pending_screenshot is not None:
                path = self._pending_screenshot
                self._pending_screenshot = None
                # Same camera state as the displayed frame: the render
                # below is background-free RGBA (transparent export).
                self._save_screenshot(
                    path, self.viewer.get_transparent_render(*self.window.dimensions)
                )
            image = (image.permute(0, 2, 3, 1).cpu().contiguous().numpy() * 255).astype(
                np.uint8
            )
            if self.overlay is not None:
                text = self.overlay(self.viewer.camera)
                if text:
                    image = draw_text_overlay(image[0], text)
            else:
                image = image[0]
            self.frame_ready.emit(image)
        except Exception:
            traceback.print_exc()
