from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def to_serializable(obj: Any) -> Any:
    """Recursively converts an object into a torch.save-serializable form.

    Tensors are cloned, detached and moved to cpu. Nested dicts, lists and
    tuples are traversed; all other values pass through unchanged.

    Args:
        obj: Object to convert.

    Returns:
        out: The serializable version of ``obj``.
    """
    if isinstance(obj, Tensor):
        return obj.clone().detach().cpu()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [to_serializable(value) for value in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    return obj


def save_dict(data: dict[str, Any], path) -> None:
    """Saves a (possibly nested) dict to file with torch.save.

    Every tensor in the dict is cloned, detached and moved to cpu before
    saving.

    Args:
        data: The dict to save.
        path: Destination file path.
    """
    torch.save(to_serializable(data), path)


def load_dict(path) -> dict[str, Any]:
    """Loads a dict saved with :func:`save_dict`.

    Args:
        path: Source file path.

    Returns:
        out: The loaded dict.
    """
    return torch.load(path)
