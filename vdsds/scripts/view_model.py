import torch

from ..load import load_model
from .script import ViewableScript


class ViewModel(ViewableScript):
    """
    Loads a model and keeps a live view of it open until the user closes the
    window.
    """

    def __init__(
        self,
        model_path: str,
        fps: int = 30,
        device: torch.device = torch.device("cuda:0"),
        view: bool = True,
    ):
        model = load_model(model_path).to(device).requires_grad_(False)
        super().__init__(model=model, fps=fps, view=view)

    def run(self):
        if self.view is None:
            return
        self._abort.wait()
