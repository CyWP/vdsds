import sys
import threading
import traceback

from PySide6.QtCore import QTimer

from ..view import View


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

    The Qt event loop is run on the main thread while :meth:`run` executes in a
    background thread, so the script can mutate the model while the view
    visualizes it. The view window closes - and ``run`` is aborted - either when
    ``run`` returns or when the user closes the window.
    """

    def __init__(self, model, fps: int = 30, view: bool = True):
        super().__init__()
        self.model = model
        self.fps = fps
        self._abort = threading.Event()
        self.view = View(model, fps=fps, on_close=self._abort.set) if view else None

    def abort_requested(self) -> bool:
        """True once the user has requested the view window to close."""
        return self._abort.is_set()

    def launch(self) -> int:
        """
        Run the script. With a view, the Qt event loop drives the main thread
        until the window closes; without one, ``run`` executes directly.
        """
        if self.view is None:
            self.run()
            return 0
        threading.Thread(target=self._run_in_thread, daemon=True).start()
        return self.view.exec()

    def _run_in_thread(self):
        try:
            self.run()
        except Exception:
            traceback.print_exc()
        finally:
            QTimer.singleShot(0, self.view.close)

    def run(self):
        raise NotImplementedError
