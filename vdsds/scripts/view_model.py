from .base import ViewableScript


class ViewModel(ViewableScript):
    """
    Loads a model and keeps a live view of it open until the user closes the
    window.
    """

    def run(self):
        """Nothing to compute; the view persists until the user closes it."""
