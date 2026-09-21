import threading
import traceback
from typing import Any, ClassVar

import torch
from easydict import EasyDict as edict
from PySide6.QtCore import QTimer

from ..deformations import get_deformation
from ..rasterizable import Rasterizable
from ..representations import load_model
from ..view import View


class Script:
    """
    Base class for runnable scripts. Subclasses implement :meth:`run`.
    """

    def run(self):
        raise NotImplementedError


class ViewableScriptConfig(edict):
    _defaults: ClassVar[dict[str, any]] = {
        "window": {
            "fps": 30,
            "view": True,
            "close_on_finish": False,
            "finish_on_close": True,
            "bg_color": [0.2, 0.2, 0.2],
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**{**self._defaults, **kwargs})


class ViewableScript(Script):
    """
    A :class:`Script` that can optionally show a live view of the model it
    operates on.

    Subclasses only implement :meth:`run` as plain script logic and never
    interact with the view directly. ``launch`` is the wrapper that spins up the
    viewer: the Qt event loop runs on the main thread while ``run`` executes in
    a background thread, so the script can mutate the model while the view
    visualizes it.
    """

    _config_defaults: ClassVar[dict[str, any]] = {
        "model": {"name": "textured_mesh", "path": None}
    }

    def __init__(
        self,
        device: torch.device,
        model_path: str,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = ViewableScriptConfig(**{**self._config_defaults, **config})
        cfg = self.config
        self.model = self.load_model().to(device)
        self._closed = threading.Event()
        if cfg.window.view:
            self.view = View(self.model, fps=cfg.window.fps, on_close=self._on_close)
            self.set_background(
                torch.tensor(
                    cfg.window.bg_color, device=self.device, dtype=torch.float32
                )
            )
        else:
            self.view = None

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def abort_requested(self) -> bool:
        """True once the user has closed the window while ``finish_on_close`` is set."""
        return self.config.window.finish_on_close and self._closed.is_set()

    def _on_close(self):
        self._closed.set()
        if self.config.window.finish_on_close:
            self.finish()

    def load_model(self) -> Rasterizable:
        cfg = self.config
        model = load_model(cfg.model.path, **cfg.model)
        if hasattr(cfg, "deformation"):
            model = get_deformation(model, **cfg.deformation)
        return model

    def launch(self) -> int:
        """
        Run the script, keeping the view (if any) alive alongside it.

        The Qt event loop drives the main thread while :meth:`run` executes in a
        background thread.

        - If ``close_on_finish`` is set, the view closes as soon as ``run``
          returns; otherwise it stays open until the user closes it.
        - If ``finish_on_close`` is set, closing the window signals the script
          to finish (check :meth:`abort_requested`); otherwise the script runs to
          completion regardless of the window state.
        """
        if self.view is None:
            self.run()
            return 0
        thread = threading.Thread(target=self._run_in_thread)
        thread.start()
        rc = self.view.exec()
        thread.join()  # don't leave the process early if closing doesn't finish the script
        return rc

    def _run_in_thread(self):
        try:
            self.run()
        except Exception:
            traceback.print_exc()
        finally:
            self._finalize()

    def _finalize(self):
        if self.view is None:
            return
        if self.config.window.close_on_finish:
            self.finish()
            QTimer.singleShot(0, self.view.close)
            return
        # Keep the window open until the user closes it.
        self._closed.wait()
        if not self.config.window.finish_on_close:
            # Natural end: the script finished and closing is not what finishes it.
            self.finish()
        QTimer.singleShot(0, self.view.close)

    def set_background(self, *args, **kwargs):
        self.view.viewer.set_background(*args, **kwargs)

    def run(self):
        raise NotImplementedError

    def finish(self):
        pass
