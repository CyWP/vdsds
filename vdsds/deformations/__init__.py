from pathlib import Path

from ..representations.base import Model
from .base import Deformation
from .jacobian_full_deform import FullJacobianDeformation
from .vd_jacobian_full_deform import VDFullJacobianDeformation
from .vd_jacobian_restrained_deform import VDRestrainedJacobianDeformation

DEFORMATION_REGISTRY: dict[str, type[Deformation]] = {
    cls.__name__: cls
    for cls in (
        FullJacobianDeformation,
        VDFullJacobianDeformation,
        VDRestrainedJacobianDeformation,
    )
}


def get_deformation(model: Model, name: str, **kwargs) -> Deformation:
    if name == "full":
        return FullJacobianDeformation(model, **kwargs)
    elif name == "vd_full":
        return VDFullJacobianDeformation(model, **kwargs)
    elif name == "vd_restrained":
        return VDRestrainedJacobianDeformation(model, **kwargs)
    else:
        raise KeyError(f"Name '{name}' is not  valid deformation type.")


def load_deformation(path: str | Path) -> Deformation:
    """Loads a saved deformation from file.

    The concrete deformation class is inferred from the "class" entry of the
    saved dict, and its model from the nested "model" dict.

    Args:
        path: Path to a ``.vd3d`` file saved via
            :meth:`Deformation.save`.

    Returns:
        out: The loaded deformation.

    Raises:
        ValueError: If the file extension is not ``.vd3d``.
    """
    if isinstance(path, str):
        path = Path(path)
    if path.suffix != ".vd3d":
        raise ValueError(f"File path '{path}' is an invalid format.")
    return Deformation.load(path)
