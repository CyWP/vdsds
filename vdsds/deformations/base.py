from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..rasterizable import Rasterizable
from ..representations.base import Model
from ..utils.camera import Camera
from ..utils.light import LightSource
from ..utils.serialization import load_dict, save_dict, to_serializable


class Deformation(nn.Module, Rasterizable):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
        return next(self.parameters()).dtype

    def _dict_data(self) -> dict[str, Any]:
        """Raw (unserialized) dict representation of this deformation.

        The model is always saved under the nested "model" key, regardless of
        whether it is trainable. Subclasses override this to add their own
        persisted tensors/values; :meth:`to_dict` serializes the result
        exactly once.

        Returns:
            out: Dict with the class name under "class" and the serialized
                model under "model".
        """
        return {
            "class": type(self).__name__,
            "model": self.model.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializes the deformation into a nested dict.

        Returns:
            out: Dict as produced by :meth:`_dict_data`, with every tensor
                cloned, detached and moved to cpu.
        """
        return to_serializable(self._dict_data())

    def save(self, path) -> None:
        """Saves the deformation to file using torch.save.

        Args:
            path: Destination file path.
        """
        save_dict(self.to_dict(), path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Deformation:
        """Reconstructs a deformation from a nested dict.

        The concrete deformation class is read from the "class" entry, and the
        model is reconstructed from its own nested dict.

        Args:
            data: Dict produced by :meth:`to_dict`.

        Returns:
            out: The reconstructed deformation.

        Raises:
            ValueError: If the dict contains no class information or the
                recorded class is not a known Deformation subclass.
        """
        from . import DEFORMATION_REGISTRY

        class_name = data.get("class")
        if class_name is None:
            raise ValueError("Cannot load deformation: dict has no 'class' entry.")
        deform_cls = DEFORMATION_REGISTRY.get(class_name)
        if deform_cls is None:
            raise ValueError(f"Unknown deformation class '{class_name}'.")
        return deform_cls._from_dict(data)

    @classmethod
    def load(cls, path) -> Deformation:
        """Loads a deformation from file saved with :meth:`save`.

        Args:
            path: Source file path.

        Returns:
            out: The loaded deformation.
        """
        return cls.from_dict(load_dict(path))

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Deformation:
        """Constructs this specific class from its dict representation.

        Args:
            data: Dict produced by :meth:`to_dict`.

        Returns:
            out: The reconstructed instance.
        """
        model = Model.from_dict(data["model"])
        return cls(model=model)

    def __len__(self) -> int:
        return len(self.model)

    def get_parameters(self) -> list[nn.Parameter]:
        """
        Parameters of the deformation, excluding the model's parameters unless
        the model itself is trainable.
        """
        model_trainable = any(p.requires_grad for p in self.model.parameters())
        return [
            param
            for name, param in self.named_parameters()
            if not name.startswith("model.") or model_trainable
        ]

    def deformed(self, camera: Camera) -> Model:
        raise NotImplementedError()

    def rasterize(self, camera: Camera, light: LightSource) -> Float[Tensor, "B 4 H W"]:
        return self.deformed(camera).rasterize(camera, light)

    def to(self, device: torch.device | str) -> Deformation:
        device = torch.device(device)
        super().to(device)
        self.model.to(device)
        return self

    def train(self, train_model: bool = False):
        self.requires_grad_(True)
        self.model.requires_grad_(False)
