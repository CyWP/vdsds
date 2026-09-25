from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from ..utils.camera import Camera
from ..utils.light import LightSource
from ..utils.serialization import load_dict, save_dict, to_serializable


class Model:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def device(self) -> torch.device:
        for v in self._tensors().values():
            return v.device
        raise ValueError(f"{self.__class__.__name__} has no tensors")

    @property
    def dtype(self) -> torch.dtype:
        for v in self._tensors().values():
            if v.is_floating_point():
                return v.dtype
        for v in self._tensors().values():
            return v.dtype
        raise ValueError(f"{self.__class__.__name__} has no tensors")

    def _tensors(self) -> dict[str, Tensor]:
        raise NotImplementedError

    def to(self, device: torch.device | str) -> Model:
        device = torch.device(device)
        mapped = {}
        for k, v in self._tensors().items():
            mapped[k] = v.to(device)
        self._apply_tensors(mapped)
        return self

    def _apply_tensors(self, tensor_dict: dict[str, Tensor]):
        raise NotImplementedError

    def requires_grad_(self, mode: bool = True) -> Model:
        for v in self._tensors().values():
            v.requires_grad_(mode)
        return self

    def parameters(self) -> list[Tensor]:
        return [v for v in self._tensors().values() if v.is_floating_point()]

    def _dict_data(self) -> dict[str, Any]:
        """Raw (unserialized) dict representation of this model.

        Subclasses override this to add/replace entries; :meth:`to_dict`
        serializes the result exactly once.

        Returns:
            out: Dict with the class name under "class" and every persisted
                tensor/value under its attribute name.
        """
        data: dict[str, Any] = {"class": type(self).__name__}
        for key, value in self._tensors().items():
            if key in data:
                continue
            data[key] = value
        return data

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model into a nested dict.

        Returns:
            out: Dict as produced by :meth:`_dict_data`, with every tensor
                cloned, detached and moved to cpu.
        """
        return to_serializable(self._dict_data())

    def save(self, path) -> None:
        """Saves the model to file using torch.save.

        Args:
            path: Destination file path.
        """
        save_dict(self.to_dict(), path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Model:
        """Reconstructs a model from a nested dict.

        The concrete model class is read from the "class" entry, so the
        returned instance may be a subclass of ``cls``.

        Args:
            data: Dict produced by :meth:`to_dict`.

        Returns:
            out: The reconstructed model.

        Raises:
            ValueError: If the dict contains no class information or the
                recorded class is not a registered Model subclass.
        """
        from . import MODEL_REGISTRY

        class_name = data.get("class")
        if class_name is None:
            raise ValueError("Cannot load model: dict has no 'class' entry.")
        model_cls = MODEL_REGISTRY.get(class_name)
        if model_cls is None:
            raise ValueError(f"Unknown model class '{class_name}'.")
        return model_cls._from_dict(data)

    @classmethod
    def load(cls, path) -> Model:
        """Loads a model from file saved with :meth:`save`.

        Args:
            path: Source file path.

        Returns:
            out: The loaded model.
        """
        return cls.from_dict(load_dict(path))

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Model:
        """Constructs this specific class from its dict representation.

        Subclasses with extra constructor arguments override this method.

        Args:
            data: Dict produced by :meth:`to_dict` (without the "class"
                entry processed).

        Returns:
            out: The reconstructed instance.
        """
        raise NotImplementedError

    def copy(self, deep: bool = False) -> Model:
        """Copies this model without re-running its constructor.

        By default the copy shares every tensor attribute with the original
        (shallow copy), preserving device, grad tracking and any autograd
        graph exactly as-is. With ``deep=True`` every tensor attribute is
        cloned instead; clones remain connected to the original tensors, so
        gradients still flow back to them.

        Args:
            deep: Whether to clone tensor attributes.

        Returns:
            out: The copy.
        """
        new = object.__new__(type(self))
        for key, value in self.__dict__.items():
            if deep and isinstance(value, Tensor):
                value = value.clone()
            new.__dict__[key] = value
        return new

    @classmethod
    def combine(cls, models: list[Model]) -> Model:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def centroid(self) -> torch.Tensor:
        raise NotImplementedError()

    def rasterize(self, camera: Camera, light: LightSource) -> Float[Tensor, "B 4 H W"]:
        raise NotImplementedError()

    def forward(self, camera: Camera, light: LightSource) -> Float[Tensor, "B 4 H W"]:
        return self.rasterize(camera, light)
