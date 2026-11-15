"""Silhouette consistency loss.

Ported from ``vd-3d/src/trainers/splatmesh_img_trainer.py``
(``SplatMeshImageTrainer.silhouette_loss``): penalises rendered vertices
that project outside the ground-truth silhouette with a top-k Chamfer
distance to the mask's own (sampled) silhouette pixels. The inverse
direction (mask pixels uncovered by the rendered mesh) was commented
out in the source -- in a degenerate vertex-to-mask form -- and is
implemented here in its meaningful mask-to-vertex direction behind the
``in_weight`` argument (default 0 = original behavior).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Float
from torch import Tensor

from ..rendering.camera import Camera
from ..representations.mesh import Mesh
from .img import ImgUtils


def silhouette_loss(
    tgt_imgs: Float[Tensor, "B 4 H W"],
    meshes: Sequence[Mesh | Float[Tensor, "V 3"]],
    cameras: Sequence[Camera],
    N_samples: int = 10000,
    alpha_threshold: float = 0.25,
    top_k: int = 10,
    out_weight: float = 10.0,
    in_weight: float = 0.0,
) -> Float[Tensor, ""]:
    """Top-k Chamfer distance from projected out-of-mask vertices to the
    target silhouette.

    Args:
        tgt_imgs: Target images (B, 4, H, W); channel 3 is the mask/alpha
            in [0, 1].
        meshes: One mesh (or vertex tensor (V, 3)) per image; V tells
            which model the deformation produced under ``cameras[i]``.
        cameras: One camera per image; must pair with ``meshes``.
        N_samples: Maximum number of silhouette pixels sampled per image.
        alpha_threshold: Alpha above which a projected vertex counts as
            inside the target silhouette.
        top_k: Number of nearest silhouette pixels averaged per outlier.
        out_weight: Weight of the out-of-mask (over-coverage) term.
        in_weight: Weight of the under-coverage (inverse) term --
            squared distance from sampled target-silhouette pixels to
            the nearest projected vertex, expanding the mesh to cover
            mask regions it does not reach. Disabled (0) in the source
            trainer and kept disabled by default; set > 0 to enable it.

    Returns:
        Scalar loss averaged over the batch (images without mask pixels
        contribute nothing).
    """
    B, _, H, W = tgt_imgs.shape
    device = tgt_imgs.device
    hw_tensor = torch.tensor([W, H], device=device).unsqueeze(0)
    mask = tgt_imgs[:, 3]

    total_loss = torch.tensor(0.0, device=device)

    for i in range(B):
        verts = meshes[i].V if isinstance(meshes[i], Mesh) else meshes[i]
        cam = cameras[i]

        image_loss = torch.tensor(0.0, device=device)

        img = mask[i]
        idx = torch.nonzero(img)

        if len(idx) == 0:
            continue

        idx = idx[torch.randperm(len(idx))[: min(len(idx), N_samples)]]
        tgt_co = torch.stack([idx[:, 1], idx[:, 0]], dim=1) / hw_tensor

        proj = cam.project_w2i(verts)  # (V, 2) = (x, y) pixels

        # uv_sample expects (row, col) pixel-center coordinates, while
        # project_w2i yields (x, y); swap so the mask is sampled at the
        # projected pixel, not at its transpose. Normalization follows
        # the swapped order (y/H, x/W), unlike the (x/W, y/H) of the
        # hw_tensor layout used for the Chamfer distances below.
        proj_rc = torch.stack([proj[:, 1], proj[:, 0]], dim=1)
        sampled = ImgUtils.uv_sample(
            img.unsqueeze(0).unsqueeze(0), proj_rc / hw_tensor.flip(-1)
        ).flatten()

        in_mask = sampled > alpha_threshold

        # Over-coverage (out) term: projected vertices outside the
        # target silhouette are pulled toward its nearest (sampled)
        # pixels, shrinking the mesh where it over-covers the target.
        if (~in_mask).any():
            sqdist = torch.cdist(proj[~in_mask] / hw_tensor, tgt_co) ** 2
            kchamfers, _ = torch.topk(
                sqdist, min(top_k, sqdist.shape[1]), dim=1, largest=False
            )
            image_loss += kchamfers.mean() * out_weight

        # Under-coverage (inverse) term: distance from the sampled
        # target-silhouette pixels to the nearest projected vertex.
        # Target regions the mesh fails to reach are far from every
        # vertex projection, so this expands the mesh to fill the
        # silhouette. (The source trainer carried this direction
        # commented out in a degenerate vertex-to-mask form: an in-mask
        # vertex is always within the mask's sampling spacing of some
        # mask pixel, making that term a shape-independent ~0.)
        if in_weight > 0.0:
            px = proj / hw_tensor  # (V, 2), (x/W, y/H)
            k = min(top_k, px.shape[0])
            kchamfers = torch.cat(
                [
                    torch.topk(
                        torch.cdist(tgt_co[j : j + 4096], px) ** 2,
                        k,
                        dim=1,
                        largest=False,
                    ).values
                    for j in range(0, tgt_co.shape[0], 4096)
                ]
            )
            image_loss += kchamfers.mean() * in_weight

        # Each term is weighted directly (out_weight / in_weight from
        # above, scaled by the config multipliers upstream); averaging
        # by the applied-term count would silently rescale the out term
        # whenever the inverse term is enabled.
        total_loss += image_loss

    return total_loss / B
