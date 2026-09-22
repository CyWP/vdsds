import threading
import traceback
from pathlib import Path
from typing import ClassVar

import torch
from PySide6.QtCore import QTimer

from ..deformations import get_deformation, load_deformation
from ..rasterizable import Rasterizable
from ..representations import get_model
from ..utils.config import Config
from ..view import View


class Script:
    """
    Base class for runnable scripts. Subclasses implement :meth:`run`.
    """

    _default_config_overrides: ClassVar[Config] = Config({})

    def __init__(
        self,
        device: torch.device,
        config: Config,
        **kwargs,
    ):
        cfg = self.default_config
        cfg.update(config)
        self.config = cfg

    def launch(self) -> None:
        """
        Run the script.
        """
        self.run()

    def run(self):
        raise NotImplementedError

    @property
    def default_config(self) -> Config:
        config = Config({})

        for cls in reversed(type(self).__mro__):
            overrides = cls.__dict__.get("_default_config_overrides")
            if overrides is not None:
                config.update(overrides)

        return config

    def load_model(self) -> Rasterizable:
        cfg = self.config
        path = Path(cfg.path.model)
        if path.suffix == ".vd3d":
            return load_deformation(path)
        model = get_model(path, **cfg.model)
        if hasattr(cfg, "deformation"):
            model = get_deformation(model, **cfg.deformation)
        return model


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

    _default_config_overrides: ClassVar[Config] = Config(
        {
            "window": {
                "fps": 30,
                "view": True,
                "close_on_finish": False,
                "finish_on_close": True,
                "bg_color": [0.2, 0.2, 0.2],
            },
            "model": {
                "name": "textured_mesh",
            },
            "path": {"model": None},
        }
    )

    def __init__(
        self,
        device: torch.device,
        config: Config,
        **kwargs,
    ):
        super().__init__(device, config)
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

    def launch(self) -> int:
        """
        Run the script, keeping the view (if any) alive alongside it.

        The Qt event loop drives the main thread while :meth:`run` executes in a
        background thread. ``finish`` is called exactly once, after ``run``
        returns.

        - If ``close_on_finish`` is set, the view closes as soon as ``run``
          returns; otherwise it stays open until the user closes it.
        - If ``finish_on_close`` is set, closing the window signals the script
          to finish (check :meth:`abort_requested`); otherwise the script runs to
          completion regardless of the window state.
        """
        if self.view is None:
            try:
                self.run()
            finally:
                self.finish()
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
        self.finish()
        QTimer.singleShot(0, self.view.close)

    def set_background(self, *args, **kwargs):
        self.view.viewer.set_background(*args, **kwargs)

    def run(self):
        raise NotImplementedError

    def finish(self):
        pass
