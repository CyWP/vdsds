from pathlib import Path

from .utils.loading import load_glb, load_obj, load_ply
from .representations.splat import Splat
from .representations.splat_mesh import SplatMesh
from .representations.textured_mesh import TexturedMesh
from .representations.vertextured_mesh import VerTexturedMesh


def load_model(path: str):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist.")
    extension = path.suffix
    if extension == ".obj":
        return SplatMesh.from_mesh_data(**load_obj(path))
    if extension == ".glb":
        return VerTexturedMesh.from_mesh_data(**load_glb(path))
        return TexturedMesh.from_mesh_data(**load_glb(path))
        return SplatMesh.from_mesh_data(**load_glb(path))
    if extension == ".ply":
        return Splat(**load_ply(path))
