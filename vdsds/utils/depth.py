"""Basic static wrapper around DepthAnythingV3 for monocular depth inference.

Monocular relative depth only: BCHW image tensors in, relative depth
tensors out. No multi-view consolidation, no camera pose estimation,
no metric scaling, no export features, no mixed input formats.

The model is never cached across calls: loading and unloading happen
inside a single inference function, so DA v3 is only resident on the
device for the duration of one inference over a set of images (e.g.
once, before a training run starts).

The ``depth_anything_3`` package is only required when inference is
actually requested; it is imported lazily in
:meth:`DepthAnything._load_model` (the package pulls a heavy dependency
chain and its pretrained checkpoint is fetched from the HuggingFace hub
on first load).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from .img import ImgUtils, Splimage
from ..rendering.shader import NO_DEPTH


class DepthAnything:
    """Monocular relative-depth wrapper over DepthAnythingV3.

    The model is loaded fresh for each inference call and unloaded
    afterwards; no state is kept between calls.
    """

    _model_id: str = "depth-anything/DA3MONO-LARGE"

    # The renderer's no-geometry sentinel ('Depth' shader in the
    # rendering module); in the loss, every value below zero counts as
    # no depth.
    no_depth: float = NO_DEPTH

    @staticmethod
    def default_device() -> torch.device:
        """Default inference device: cuda when available, else cpu."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def _load_model(cls, device: torch.device | str | None = None):
        """Load the pretrained checkpoint as a fresh instance.

        Args:
            device: Target inference device; defaults to
                :meth:`default_device`.

        Returns:
            out: The loaded DA v3 model on the target device.
        """
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError(
                "Monocular depth inference requires the "
                "'depth_anything_3' package (pip install depth_anything_3)."
            ) from e
        device = torch.device(device) if isinstance(device, str) else device
        return DepthAnything3.from_pretrained(cls._model_id).to(
            device or cls.default_device()
        )

    @classmethod
    def _run_inference(
        cls,
        inputs: Sequence[Image.Image | str],
        device: torch.device | str | None = None,
    ) -> list[Float[Tensor, "H W"]]:
        """Single load-infer-unload cycle over a set of images.

        The model is loaded once for the whole set, every image is
        inferred individually (their processed resolutions may differ),
        and the model is unloaded afterwards.

        Note: each prediction also carries a confidence map
        (``prediction.conf``, (N, H, W)); it is unused here but could
        be interesting for weighting the depth loss.

        Args:
            inputs: One DA v3 inference input per image (PIL images or
                paths).
            device: Inference device; defaults to
                :meth:`default_device`.

        Returns:
            out: One relative depth (H, W) float32 CPU tensor per
                input, in input order.
        """
        model = cls._load_model(device=device)
        try:
            depths = []
            for inp in inputs:
                prediction = model.inference([inp])
                depths.append(
                    torch.from_numpy(np.asarray(prediction.depth))
                    .squeeze(0)
                    .float()
                )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return depths

    @classmethod
    def depth(
        cls,
        imgs: Float[Tensor, "B C H W"],
        device: torch.device | str | None = None,
    ) -> Float[Tensor, "B 1 H W"]:
        """Run basic monocular relative depth inference on image tensors.

        Args:
            imgs: Image batch (B, C, H, W) in [0, 1]. Channels follow the
                repo's Splimage conventions: gray (1) is expanded to RGB,
                alpha (4) is dropped, and RGB (3) is used as-is.
            device: Inference device; defaults to :meth:`default_device`.

        Returns:
            out: Relative depth (B, 1, H, W) float32 tensor on CPU, at
                the model's processed input resolution.
        """
        assert isinstance(imgs, Tensor) and imgs.ndim == 4, (
            "Monocular depth expects a BCHW tensor (B, C, H, W); "
            f"got shape {tuple(imgs.shape)}."
        )
        inputs = [cls._as_inference_input(img) for img in imgs]
        depths = cls._run_inference(inputs, device=device)
        return torch.stack(depths).unsqueeze(1)

    @classmethod
    def _as_inference_input(cls, img: Float[Tensor, "C H W"]) -> Image.Image:
        """Normalize a single-channel-last image tensor to an RGB PIL input.

        Channels follow the Splimage conventions: gray is expanded to
        RGB, RGBA drops its alpha, RGB is used as-is.
        """
        spl = Splimage(img[None])
        rgb = spl.image()
        C = rgb.shape[1]
        if C == 1:
            rgb = rgb.repeat(1, 3, 1, 1)
        elif C == 4:
            rgb = rgb[:, :3]
        return ImgUtils.tensor2pil(rgb)

    @classmethod
    def infer_depths(
        cls,
        imgs: Sequence[Float[Tensor, "C H W"]],
        device: torch.device | str | None = None,
    ) -> list[Float[Tensor, "H W"]]:
        """Monocular relative depth for a set of images.

        A single load-infer-unload cycle covers the whole set (see
        :meth:`_run_inference`); images are inferred individually, so
        their processed resolutions may differ.

        Args:
            imgs: One (C, H, W) image tensor per entry, following the
                Splimage channel conventions.
            device: Inference device; defaults to
                :meth:`default_device`.

        Returns:
            out: One relative depth (H, W) float32 CPU tensor per
                input, in input order.
        """
        inputs = [cls._as_inference_input(img) for img in imgs]
        return cls._run_inference(inputs, device=device)

    @classmethod
    def depth_from_constraints(
        cls, constraints, device: torch.device | str | None = None
    ) -> dict[str, Float[Tensor, "H W"]]:
        """Monocular relative depth of all images in the constraints.

        Constraints always carry views; only the image is optional.
        Overlays are used through their viewport placement
        (``Overlay.image`` at the constraint camera's resolution) --
        that placement is the point of aligning the images, so it is
        never bypassed.

        Args:
            constraints: The id-keyed constraint dict.
            device: Inference device; defaults to
                :meth:`default_device`.

        Returns:
            out: cid -> relative depth (H, W) float32 CPU tensor, at
                the model's processed resolution for that image.
        """
        ids, imgs = [], []
        for cid, constraint in constraints.items():
            view = constraint.view
            overlay = view.overlay if view is not None else None
            if overlay is None:
                continue
            camera = view.camera
            placed = overlay.image(camera.H, camera.W)  # (1, 4, H, W)
            ids.append(cid)
            imgs.append(placed[0])
        if not ids:
            return {}
        depths = cls.infer_depths(imgs, device=device)
        return dict(zip(ids, depths))


def _minmax(x: Float[Tensor, "N"]) -> Float[Tensor, "N"]:
    """Range-normalize to [0, 1] over the map (denominator clamped)."""
    return (x - x.amin()) / (x.amax() - x.amin()).clamp_min(1e-6)


def depth_loss(
    rendered: Float[Tensor, "B 1 H W"],
    measured: Float[Tensor, "B 1 H W"],
    mode: str = "affine",  # Options: 'affine', 'minmax'
    reduction: str = "mean",  # Options: 'mean', 'none'
) -> Float[Tensor, " "] | Float[Tensor, " B"]:
    """Depth loss matching the ranges/distributions of two depth maps.

    Metric depth is meaningless for rendered-vs-monocular
    comparison, so the loss matches distributions only: per image,
    the monocular relative depths are fitted to the rendered
    clip-space depths (closed-form affine alignment in 'affine'
    mode, per-map range normalization in 'minmax' mode) and only
    the residuals are penalized.

    Pixels below zero in the rendered map (which includes the
    renderer's ``NO_DEPTH`` background sentinel) are treated as
    no-geometry and are excluded from the loss entirely; their
    monocular values are garbage and must not pull the fit.

    Args:
        rendered: Rendered depth maps (B, 1, H, W) from
            ``Renderer.depth()``: valid values in (0, 1], background
            at ``NO_DEPTH`` (-1).
        measured: Monocular relative depths (B, 1, H, W) from
            ``DepthAnything.depth`` (resampled to the render
            resolution).
        mode: 'affine' (scale-and-shift-invariant least squares) or
            'minmax' (per-image range normalization; does NOT
            handle inverted orderings, use 'affine' for that).
        reduction: 'mean' over batch images or 'none' for a
            per-image tensor. Images without rendered geometry
            contribute zero and are skipped in the mean.

    Returns:
        out: Scalar loss with 'mean' reduction (0 when no image has
            rendered geometry), else the per-image losses stacked
            (N,) for images with geometry (an empty tensor when
            none).
    """
    if mode not in ("affine", "minmax"):
        raise ValueError(f"Depth-loss mode '{mode}' is invalid.")
    B = rendered.shape[0]
    r = rendered[:, 0].reshape(B, -1)
    m = measured[:, 0].reshape(B, -1)
    # Below zero = no depth (renderer background sentinel): excluded.
    valid = (rendered[:, 0] >= 0.0).reshape(B, -1)
    losses = []
    for b in range(B):
        rb, mb = r[b][valid[b]], m[b][valid[b]]
        if rb.numel() == 0:
            continue  # no rendered geometry: no loss
        if mode == "affine":
            # Closed-form least squares fit measured ~ a * rendered + b;
            # a may be negative (monocular depth often encodes
            # nearness, clip-space depth encodes distance).
            mean_r, mean_m = rb.mean(), mb.mean()
            var_r = ((rb - mean_r) ** 2).mean()
            if var_r > 0:
                cov = ((rb - mean_r) * (mb - mean_m)).mean()
                a = cov / var_r
                residual = a * rb + (mean_m - a * mean_r) - mb
            else:
                residual = mb - mean_m  # constant render: shift only
            losses.append(residual.abs().mean())
        elif mode == "minmax":
            losses.append((_minmax(rb) - _minmax(mb)).abs().mean())
    if len(losses) == 0:
        return torch.zeros(()) if reduction == "mean" else torch.zeros(0)
    stacked = torch.stack(losses)
    return stacked.mean() if reduction == "mean" else stacked


# The silhouette loss lives in ``vdsds/utils/silhouette.py`` (original
# port of ``SplatMeshImageTrainer.silhouette_loss``, top-k Chamfer
# distance from projected out-of-mask vertices to the target-silhouette
# pixels); it must not be reimplemented here.

    # ------------------------------------------------------------------
    # Depth loss
    # ------------------------------------------------------------------
    # (Implemented as the module-level `depth_loss` below, kept outside
    # the class: it is a pure function of the two depth maps and needs
    # no wrapper state.)

