from pathlib import Path

from .representations.textured_mesh import TexturedMesh
from .representations.vertextured_mesh import VerTexturedMesh
from .utils.loading import load_glb, load_obj


def load_model(path: str):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist.")
    extension = path.suffix
    if extension == ".obj":
        return TexturedMesh.from_mesh_data(**load_obj(path))
    if extension == ".glb":
        # return Mesh.from_mesh_data(**load_glb(path))
        return VerTexturedMesh.from_mesh_data(**load_glb(path))
        return TexturedMesh.from_mesh_data(**load_glb(path))
