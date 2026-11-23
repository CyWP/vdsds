from pathlib import Path, PureWindowsPath

import torch
import trimesh
from jaxtyping import Shaped
from torch import Tensor

from .img import Splimage


def _normalize_uv(uv: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Folds loader UVs into the [0, 1] x [0, 1] sampling range.

    Per axis, in order:
    - range within [0, 1]: kept (the standard convention).
    - range within [-1, 0]: negated. Some assets (e.g. retextured
      exports) author v = -v_std against the image as stored; wrapping
      such UVs would mirror the texture instead (verified visually).
    - otherwise: wrapped by integer period (``torch.remainder``), the
      repeat/atlas case.

    Args:
        uv: UV coordinates (N, 2).

    Returns:
        out: UV coordinates (N, 2) with both axes in [0, 1].
    """
    out = uv.clone()
    for axis in (0, 1):
        vals = uv[:, axis]
        lo, hi = vals.min(), vals.max()
        if lo >= -eps and hi <= 1 + eps:
            continue
        if lo >= -1 - eps and hi <= eps:
            out[:, axis] = -vals
        else:
            out[:, axis] = torch.remainder(vals, 1.0)
    return out


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
                    if not isinstance(tex_file, PureWindowsPath):
                        # mtl paths are relative to the obj directory;
                        # an absolute path is kept as-is by the join.
                        tex_file = path.parent / tex_file
                    if isinstance(tex_file, PureWindowsPath) or not tex_file.exists():
                        tex_file = path.parent / "textures" / tex_file.name
                        if not tex_file.exists():
                            tex_file = path.parent / "Textures" / tex_file.name
                        if not tex_file.exists():
                            raise FileNotFoundError(
                                f"Could not find albedo texture file for {path}."
                            )
                    mtl = Splimage(tex_file, force_rgba=False)._tensor[:, :3]

        except Exception as e:
            raise e
        uv_co = torch.tensor(uv_co)[:, :2]
        # OBJ 'vt' v runs bottom-up (v=0 at the image bottom); loaders
        # standardize uv_co to (u, v), top-down image space (matching
        # dr.texture and the Splimage samplers), so v is flipped here.
        uv_co[:, 1] = 1 - uv_co[:, 1]
        uv_co = _normalize_uv(uv_co)
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
        # glTF UVs are (u, v) with v top-down already; they are kept
        # native. Loaders standardize uv_co to (u, v), top-down image
        # space: dr.texture samples v top-down (row 0 = image top), and
        # Splimage.image_sample consumers (e.g. the VerTexturedMesh UV
        # bake) transpose to their (row, col) convention at the call
        # site. The v flip here produced vertically mirrored textures.
        # uv_co = torch.stack([1 - uv_co[:, 1], 1 - uv_co[:, 0]], dim=1)
        uv_idx = F.clone()
        mtl = Splimage(mesh.visual.material.baseColorTexture, force_rgba=False)._tensor[
            :, :3
        ]
    if uv_co is not None:
        uv_co = _normalize_uv(uv_co)
    # V[:, 2] *= -1
    V = -torch.stack([V[:, 2], V[:, 0], V[:, 1]], dim=1)
    return {
        "V": V,
        "F": F,
        "uv_co": uv_co,
        "uv_idx": uv_idx,
        "texture": mtl,
    }
