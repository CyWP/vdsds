import numpy as np
import torch
import trimesh
from PIL import Image


def _quad_image():
    """16x16 RGBA texture: red/green top, blue/yellow bottom."""
    img = np.zeros((16, 16, 4), np.uint8)
    img[:8, :8] = (255, 0, 0, 255)
    img[:8, 8:] = (0, 255, 0, 255)
    img[8:, :8] = (0, 0, 255, 255)
    img[8:, 8:] = (255, 255, 0, 255)
    return img


def _corner_uvs():
    """glTF-native UVs: v is top-down (v=0 image top)."""
    return np.array(
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]], np.float32
    )


def _write_glb(tmp_path, uv):
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(V, F, process=False)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv, image=Image.fromarray(_quad_image(), "RGBA")
    )
    path = tmp_path / "uvtest.glb"
    path.write_bytes(trimesh.exchange.export.export_glb(mesh))
    return path


def _bake(data):
    from vdsds.representations.vertextured_mesh import VerTexturedMesh

    return VerTexturedMesh.from_mesh_data(**data).texture


def test_load_glb_keeps_native_topdown_uv(tmp_path):
    """glTF UVs are (u, v) top-down already; the loader must not flip
    v (a 1-v flip mirrored textures vertically)."""
    from vdsds.utils.loading import load_glb

    uv = _corner_uvs()
    data = load_glb(_write_glb(tmp_path, uv))
    assert torch.allclose(
        data["uv_co"], torch.tensor(uv), atol=1e-5
    )


def test_vertextured_bake_matches_image_space(tmp_path):
    """The per-vertex UV bake samples the texture in image space: the
    vertex at (u=0.25, v=0.25) gets the top-left color, etc. (no
    transpose, no flip)."""
    from vdsds.utils.loading import load_glb

    data = load_glb(_write_glb(tmp_path, _corner_uvs()))
    colors = _bake(data)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    assert torch.allclose(colors, expected, atol=1e-4)


def test_load_obj_flips_v_to_topdown(tmp_path):
    """OBJ 'vt' v runs bottom-up; the loader standardizes uv_co to
    (u, v) top-down, so the baked colors match the image again."""
    from vdsds.utils.loading import load_obj

    d = tmp_path
    Image.fromarray(_quad_image(), "RGBA").save(d / "albedo.png")
    (d / "test.mtl").write_text("newmtl m\nKd 1 1 1\nmap_Kd albedo.png\n")
    # v=0.2 (bottom of the image in OBJ convention) -> baked blue after
    # the flip; u=0.25 -> left half.
    (d / "test.obj").write_text(
        "mtllib test.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0.25 0.2\nvt 0.75 0.2\nvt 0.25 0.8\nvt 0.75 0.8\n"
        "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
    )
    data = load_obj(d / "test.obj")
    # Standardized: u unchanged, v flipped to top-down.
    expected_uv = torch.tensor(
        [[0.25, 0.8], [0.75, 0.8], [0.25, 0.2], [0.75, 0.2]]
    )
    assert torch.allclose(data["uv_co"], expected_uv, atol=1e-5)
    colors = _bake(data)
    # OBJ v=0.2 is the image bottom (blue/yellow); after the loader's
    # flip those vertices bake the bottom-row colors.
    expected = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    assert torch.allclose(colors, expected, atol=1e-4)
