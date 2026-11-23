from pathlib import Path
from typing import Any

from ..utils.loading import _apply_uv_convention, load_glb, load_obj
from .base import Model
from .mesh import Mesh
from .textured_mesh import TexturedMesh
from .vertextured_mesh import VerTexturedMesh

MODEL_REGISTRY: dict[str, type[Model]] = {
    cls.__name__: cls for cls in (Mesh, TexturedMesh, VerTexturedMesh)
}


def get_model(
    path: str | Path | dict[str, Any],
    name: str = "textured_mesh",
    uv_convention: str = "native",
    **kwargs,
) -> Model:
    if isinstance(path, dict):
        if "class" in path:
            return Model.from_dict(path)
        data = path
    else:
        if isinstance(path, str):
            path = Path(path)
        extension = path.suffix
        if extension == ".m3d":
            return Model.load(path)
        elif extension == ".obj":
            data = load_obj(path)
        elif extension == ".glb":
            data = load_glb(path)
        else:
            raise ValueError(f"File path '{path}' is an invalid format.")
        # Per-asset UV convention override (see _apply_uv_convention):
        # exporter conventions are not detectable from the data, so
        # assets that disagree with the loaders' standardized (u, v)
        # top-down coordinates are corrected explicitly here.
        if data.get("uv_co") is not None:
            data["uv_co"] = _apply_uv_convention(data["uv_co"], uv_convention)
    if name == "mesh":
        return Mesh.from_mesh_data(**data, **kwargs)
    elif name == "textured_mesh":
        return TexturedMesh.from_mesh_data(**data, **kwargs)
    elif name == "vertextured_mesh":
        return VerTexturedMesh.from_mesh_data(**data, **kwargs)
    else:
        raise KeyError(f"Name '{name}' does not refer to any valid Model class.")
