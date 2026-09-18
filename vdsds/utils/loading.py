from pathlib import Path, PureWindowsPath

import torch
import trimesh
from jaxtyping import Shaped
from torch import Tensor

from .img import Splimage


def load_obj(path: Path) -> dict[str, Shaped[Tensor, "..."]]:
    assert path.suffix == ".obj"
    content = {}
    F = []
    uv_co = []
    uv_idx = []
    mtl = None
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        if not words:
            continue
        key = words[0]
        val = words[1:]
        if key in content:
            content[key].append(val)
        else:
            content[key] = [val]
    V = torch.tensor([[float(val) for val in entry] for entry in content["v"]])
    uv_co = [[float(val) for val in entry] for entry in content["vt"]]
    uv_idx = []
    for entry in content["f"]:
        for i in range(len(entry) - 2):
            face = []
            uv_face = []
            for val in [entry[0]] + entry[i + 1 : i + 3]:
                spl = val.split("/")
                face.append(int(spl[0]) - 1)
                if len(spl) > 1:
                    uv_face.append(int(spl[1]) - 1)
            F.append(face)
            if uv_face:
                uv_idx.append(uv_face)
    if len(uv_co):
        try:
            mtl_path = path.parent / content["mtllib"][0][0]
            with open(mtl_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                words = line.strip().split()
                if words and words[0] == "map_Kd":
                    s = words[1]
                    tex_file = PureWindowsPath(s) if s[1] == ":" else Path(s)
                    if isinstance(tex_file, PureWindowsPath) or not tex_file.exists():
                        tex_file = path.parent / "textures" / tex_file.name
                        if not tex_file.exists():
                            tex_file = path.parent / "Textures" / tex_file.name
                        if not tex_file.exists():
                            raise FileNotFoundError(
                                f"Could not find albedo texture file for {path}."
                            )
                    mtl = Splimage(tex_file)._tensor

        except Exception as e:
            raise e
        uv_co = torch.tensor(uv_co)[:, :2]
        uv_co[:, 1] = 1 - uv_co[:, 1]
        V[:, 1] = -V[:, 1]
    return {
        "V": V,
        "F": torch.tensor(F, dtype=torch.long),
        "uv_co": uv_co,
        "uv_idx": torch.tensor(uv_idx, dtype=torch.long),
        "texture": mtl,
    }


def load_glb(path: Path) -> dict[str, Shaped[Tensor, "..."]]:
    assert path.suffix == ".glb"
    mesh = trimesh.load(path, force="mesh")
    # mesh = mesh.subdivide(iterations=1)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.process(validate=True)
    mesh.remove_infinite_values()
    trimesh.repair.broken_faces(mesh)
    mesh.update_faces(mesh.nondegenerate_faces())

    # 2. Merge vertices that are visually identical but disconnected
    mesh.fill_holes()

    # 3. Re-run normal fixes
    mesh.fix_normals(multibody=True)
    # mesh = list(scene.geometry.values())[0]
    # meshes = list(scene.geometry.values())
    # mesh = trimesh.util.concatenate(meshes)
    V = torch.tensor(mesh.vertices, dtype=torch.float32)
    F = torch.tensor(mesh.faces, dtype=torch.long)
    uv_co = None
    uv_idx = None
    mtl = None
    if mesh.visual.kind == "texture" and mesh.visual.uv is not None:
        uv_co = torch.tensor(mesh.visual.uv, dtype=torch.float32)
        uv_co[:, 1] = 1 - uv_co[:, 1]
        # uv_co = torch.stack([1 - uv_co[:, 1], 1 - uv_co[:, 0]], dim=1)
        uv_idx = F.clone()
        mtl = Splimage(mesh.visual.material.baseColorTexture)._tensor
    # V[:, 2] *= -1
    V = torch.stack([V[:, 2], V[:, 0], -V[:, 1]], dim=1)
    return {
        "V": V,
        "F": F,
        "uv_co": uv_co,
        "uv_idx": uv_idx,
        "texture": mtl,
    }
