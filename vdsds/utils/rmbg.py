"""Basic static wrapper around BRIA RMBG-1.4 for background removal.

Foreground matting only: BCHW image tensors in, RGBA tensors out (the
input RGB kept as-is, the alpha channel carrying the foreground matte).
Used to strip photographic backgrounds from Image-constraint reference
images so the paired-noise DDS loss (and the viewport overlay) sees
only the subject over the render background color.

The model is never cached across calls: loading and unloading happen
inside a single inference function, so RMBG is only resident on the
device for the duration of one inference over a set of images. It is
imported lazily together with the checkpoint download (the weights are
fetched from the HuggingFace hub on first load and cached there).

License note: RMBG-1.4's weights are released under the BRIA
non-commercial license; use accordingly. The model id is a class
attribute, so a different (e.g. commercially licensed) checkpoint can
be swapped in via code/config without touching the call sites.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class BackgroundRemover:
    """Foreground matte extraction over RMBG-1.4.

    The model is loaded fresh for each inference call and unloaded
    afterwards; no state is kept between calls.
    """

    # BRIA RMBG-1.4 (loaded through remote code, supported by the
    # installed transformers). Swap the id to change checkpoints.
    _model_id: str = "briaai/RMBG-1.4"

    # The model's fixed preprocessing: square resize, [0, 1] scaling,
    # ImageNet mean and unit std (the checkpoint's documented input
    # convention).
    _image_size: tuple[int, int] = (1024, 1024)
    _mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    _std: tuple[float, ...] = (1.0, 1.0, 1.0)

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
            out: The loaded RMBG model on the target device.
        """
        from transformers import AutoModelForImageSegmentation

        device = torch.device(device) if isinstance(device, str) else device
        model = AutoModelForImageSegmentation.from_pretrained(
            cls._model_id, trust_remote_code=True
        )
        return model.eval().to(device or cls.default_device())

    @classmethod
    def _run_mattes(
        cls,
        imgs: Float[Tensor, "B 3 H W"],
        device: torch.device | str | None = None,
    ) -> list[Float[Tensor, "H W"]]:
        """Single load-infer-unload cycle over a set of RGB images.

        The model is loaded once for the whole set, every image is
        matted individually (resized back to its own input resolution),
        and the model is unloaded afterwards.

        Args:
            imgs: RGB image batch (B, 3, H, W) in [0, 1].
            device: Inference device; defaults to :meth:`default_device`.

        Returns:
            out: One foreground matte (H, W) float32 CPU tensor per
                input, in [0, 1], in input order.
        """
        model = cls._load_model(device=device)
        try:
            device = torch.device(device) if isinstance(device, str) else (
                device or cls.default_device()
            )
            mattes = []
            with torch.no_grad():
                for img in imgs:
                    h, w = img.shape[-2:]
                    x = F.interpolate(
                        img[None].to(device),
                        size=cls._image_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    x = (x - torch.tensor(cls._mean, device=device)[None, :, None, None]) / (
                        torch.tensor(cls._std, device=device)[None, :, None, None]
                    )
                    out = model(x)
                    # The remote-code model returns a list/tuple of
                    # logit maps; the first entry carries the matte.
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if out.shape[1] != 1:
                        out = out[:, :1]
                    matte = F.interpolate(
                        out, size=(h, w), mode="bilinear", align_corners=False
                    )[0, 0]
                    mattes.append(matte.clamp(0.0, 1.0).float().cpu())
            return mattes
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def remove(
        cls,
        imgs: Float[Tensor, "B C H W"],
        device: torch.device | str | None = None,
    ) -> Float[Tensor, "B 4 H W"]:
        """Apply foreground matting to a batch of images.

        Args:
            imgs: Image batch (B, C, H, W) in [0, 1]. Channels follow
                the repo's Splimage conventions: gray (1) is expanded
                to RGB, RGBA (4) keeps its RGB and gets a fresh matte
                as alpha, RGB (3) is used as-is.
            device: Inference device; defaults to :meth:`default_device`.

        Returns:
            out: RGBA batch (B, 4, H, W) float32 CPU tensor: the input
                RGB channels with the foreground matte as alpha.
        """
        assert isinstance(imgs, Tensor) and imgs.ndim == 4, (
            "Background removal expects a BCHW tensor (B, C, H, W); "
            f"got shape {tuple(imgs.shape)}."
        )
        spl = imgs[:, :3] if imgs.shape[1] >= 3 else imgs.repeat(1, 3, 1, 1)
        mattes = cls._run_mattes(spl, device=device)
        alpha = torch.stack(mattes).unsqueeze(1)  # (B, 1, H, W)
        return torch.cat([spl, alpha], dim=1)
