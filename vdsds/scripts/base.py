import threading
import traceback
import torch

from PySide6.QtCore import QTimer

from ..view import View
from ..load import load_model
from ..rasterizable import Rasterizable


class Script:
    """
    Base class for runnable scripts. Subclasses implement :meth:`run`.
    """

    def run(self):
        raise NotImplementedError


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

    def __init__(
        self,
        model_path: str,
        fps: int = 30,
        view: bool = True,
        close_on_finish: bool = False,
        finish_on_close: bool = True,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super().__init__()
        self.model = self.load_model(model_path).to(device)
        self.fps = fps
        self.close_on_finish = close_on_finish
        self.finish_on_close = finish_on_close
        self._closed = threading.Event()
        if view:
            self.view = View(self.model, fps=fps, on_close=self._on_close)
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
        return self.finish_on_close and self._closed.is_set()

    def _on_close(self):
        self._closed.set()
        if self.finish_on_close:
            self.finish()

    def load_model(self, path: str) -> Rasterizable:
        return load_model(path)

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
        if self.close_on_finish:
            self.finish()
            QTimer.singleShot(0, self.view.close)
            return
        # Keep the window open until the user closes it.
        self._closed.wait()
        if not self.finish_on_close:
            # Natural end: the script finished and closing is not what finishes it.
            self.finish()
        QTimer.singleShot(0, self.view.close)

    def run(self):
        raise NotImplementedError

    def finish(self):
        pass
