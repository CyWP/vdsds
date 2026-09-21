from pathlib import Path

from ..utils.loading import load_glb, load_obj
from .base import Model
from .mesh import Mesh
from .textured_mesh import TexturedMesh
from .vertextured_mesh import VerTexturedMesh


def load_model(self, path: Path, name: str, **kwargs) -> Model:
    extension = path.suffix
    if extension == ".obj":
        data = load_obj(path)
    elif extension == ".glb":
        data = load_glb(path)
    else:
        raise ValueError(f"File path '{path}' is an invalid format.")
    if name == "mesh":
        return Mesh.from_mesh_data(**data, **kwargs)
    elif name == "textured_mesh":
        return TexturedMesh.from_mesh_data(**data, **kwargs)
    elif name == "vertextured_mesh":
        return VerTexturedMesh.from_mesh_data(**data, **kwargs)
    else:
        raise KeyError(f"Name '{name}' does not refer to any valid Model class.")
