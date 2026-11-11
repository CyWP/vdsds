"""Silhouette consistency loss.

Ported from ``vd-3d/src/trainers/splatmesh_img_trainer.py``
(``SplatMeshImageTrainer.silhouette_loss``): penalises rendered vertices
that project outside the ground-truth silhouette with a top-k Chamfer
distance to the mask's own (sampled) silhouette pixels. The inverse
direction (mask pixels uncovered by the rendered mesh) was commented
out in the source; it is available here behind the ``in_weight``
argument (default 0 = original behavior).
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
            distance from projected in-mask vertices to the target
            silhouette. Disabled (0) in the source trainer and kept
            disabled by default; set > 0 to enable it.

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
        image_count = 0

        img = mask[i]
        idx = torch.nonzero(img)

        if len(idx) == 0:
            continue

        idx = idx[torch.randperm(len(idx))[: min(len(idx), N_samples)]]
        tgt_co = torch.stack([idx[:, 1], idx[:, 0]], dim=1) / hw_tensor

        proj = cam.project_w2i(verts)

        sampled = ImgUtils.uv_sample(
            img.unsqueeze(0).unsqueeze(0), proj / hw_tensor
        ).flatten()

        in_mask = sampled > alpha_threshold

        # Under-coverage (inverse) term: distance from projected
        # in-mask vertices to the target silhouette pixels. Commented
        # out in the source trainer; enabled only when 'in_weight' > 0.
        if in_weight > 0.0 and in_mask.any():
            sqdist = torch.cdist(proj[in_mask] / hw_tensor, tgt_co) ** 2
            kchamfers, _ = torch.topk(
                sqdist, min(top_k, sqdist.shape[1]), dim=1, largest=False
            )
            image_loss += kchamfers.mean() * in_weight
            image_count += 1

        if (~in_mask).any():
            sqdist = torch.cdist(proj[~in_mask] / hw_tensor, tgt_co) ** 2
            kchamfers, _ = torch.topk(
                sqdist, min(top_k, sqdist.shape[1]), dim=1, largest=False
            )
            image_loss += kchamfers.mean() * out_weight
            image_count += 1

        if image_count > 0:
            total_loss += image_loss / image_count

    return total_loss / B
