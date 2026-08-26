import torch
import numpy as np
import math
import trimesh

from pathlib import Path, PureWindowsPath
from torch import Tensor
from jaxtyping import Shaped
from typing import Dict, Tuple
from plyfile import PlyData

from .img import Splimage


def load_obj(path: Path) -> Dict[str, Shaped[Tensor, "..."]]:
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
                    mtl = Splimage(tex_file)

        except Exception as e:
            raise e
        uv_co = torch.tensor(uv_co)[:, :2]
        uv_co[:, 1] = 1 - uv_co[:, 1]
        V[:, 1] *= -1
    return {
        "V": V,
        "F": torch.tensor(F, dtype=torch.long),
        "uv_co": uv_co,
        "uv_idx": torch.tensor(uv_idx, dtype=torch.long),
        "texture": mtl,
    }


def load_glb(path: Path) -> Dict[str, Shaped[Tensor, "..."]]:
    assert path.suffix == ".glb"
    scene = trimesh.load(path, force="scene")
    mesh = list(scene.geometry.values())[0]
    V = torch.tensor(mesh.vertices, dtype=torch.float32)
    F = torch.tensor(mesh.faces, dtype=torch.long)
    uv_co = None
    uv_idx = None
    mtl = None
    if mesh.visual.kind == "texture" and mesh.visual.uv is not None:
        uv_co = torch.tensor(mesh.visual.uv, dtype=torch.float32)
        uv_co[:, 1] = 1 - uv_co[:, 1]
        uv_idx = F.clone()
        # V[:, 1] *= -1
        mtl = Splimage(mesh.visual.material.baseColorTexture)
    return {
        "V": V,
        "F": F,
        "uv_co": uv_co,
        "uv_idx": uv_idx,
        "texture": mtl,
    }


def load_ply(path: Path) -> Tuple[np.ndarray]:
    assert path.suffix == ".ply"
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    sh_degree = 3 * (len(extra_f_names) + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
    )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    means = torch.tensor(xyz, dtype=torch.float32).contiguous()
    sh0 = torch.tensor(features_dc, dtype=torch.float32).transpose(1, 2).contiguous()
    shN = torch.tensor(features_extra, dtype=torch.float32).transpose(1, 2).contiguous()
    opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float32)).squeeze()
    scales = torch.exp(torch.tensor(scales, dtype=torch.float32)).contiguous()
    Q = torch.nn.functional.normalize(
        torch.tensor(rots, dtype=torch.float32)
    ).contiguous()

    # Normalization of the colors
    max_values_per_channel, _ = torch.max(sh0, dim=-1, keepdim=True)
    max_values_per_channel = torch.clamp(
        max_values_per_channel, min=1.0
    )  # Prevent division by 0
    sh0 /= max_values_per_channel

    colors = torch.cat((sh0, shN), dim=1)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    return means - torch.mean(means, dim=0), Q, scales, opacities, colors, sh_degree
