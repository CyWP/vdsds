from ..representations.base import Model
from .base import Deformation
from .jacobian_full_deform import FullJacobianDeformation
from .vd_jacobian_full_deform import VDFullJacobianDeformation
from .vd_jacobian_restrained_deform import VDRestrainedJacobianDeformation


def get_deformation(model: Model, name: str, **kwargs) -> Deformation:
    if name == "full":
        return FullJacobianDeformation(model, **kwargs)
    elif name == "vd_full":
        return VDFullJacobianDeformation(model, **kwargs)
    elif name == "vd_restrained":
        return VDRestrainedJacobianDeformation(model, **kwargs)
    else:
        raise KeyError(f"Name '{name}' is not  valid deformation type.")
