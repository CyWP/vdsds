from __future__ import annotations

import math
import os
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from PIL import Image
from torch import Tensor


MaskMode = Literal["R", "G", "B", "A", "RGB", "mean"]


class ImgUtils:
    """Image processing utilities for tensor operations.

    Static methods for converting between image and tensor formats,
    patch extraction/assembly, Gaussian kernels, and SSIM computation.
    """

    @staticmethod
    def img2tensor(
        img: Float[Tensor, "B H W C"],
    ) -> Float[Tensor, "B C H W"]:
        """Convert [0,1] HWC image to CHW tensor.

        Args:
            img: Image tensor (B, H, W, C) in [0, 1].

        Returns:
            Tensor (B, C, H, W) in [0, 1].
        """
        return img.permute(0, 3, 1, 2)

    @staticmethod
    def tensor2img(
        x: Float[Tensor, "B C H W"],
        clamp: bool = False,
        mode: str = "RGBA",
    ) -> Float[Tensor, "B H W C"]:
        """Convert [0,1] CHW tensor to HWC image.

        Args:
            x: Tensor (B, C, H, W) in [0, 1].

        Returns:
            Image (B, H, W, C) in [0, 1].
        """
        img = x.permute(0, 2, 3, 1)
        if clamp:
            img = img.clamp(0, 1)
        return img

    @staticmethod
    def tensor2pil(
        x: Float[Tensor, "B C H W"],
    ) -> Image.Image | List[Image.Image]:
        """Convert tensor to PIL Image.

        Args:
            x: Tensor (B, C, H, W) in [0, 1].

        Returns:
            PIL Image as uint8 [0, 255].
        """
        B, C, H, W = x.shape
        mode = "RGB" if C == 3 else "RGBA"
        img = ImgUtils.tensor2img(x, clamp=True)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        imgs = []
        for i in range(B):
            imgs.append(Image.fromarray(img_np[i], mode=mode))
        if B == 1:
            return imgs[0]
        return imgs

    @staticmethod
    def pil2tensor(
        img: Image.Image | Sequence[Image.Image],
    ) -> Float[Tensor, "B C H W"]:
        """Convert PIL Image to tensor.

        Args:
            img: PIL Image or sequence of PIL Images as uint8 [0, 255].

        Returns:
            Tensor (B, C, H, W) in [0, 1].
        """
        if isinstance(img, Image.Image):
            imgs = [img]
        else:
            imgs = list(img)
        arrs = [np.asarray(im.convert("RGBA")) for im in imgs]
        stacked = np.stack(arrs, axis=0)
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0

    @staticmethod
    def pil2mask(
        img: Image.Image | Sequence[Image.Image],
        mode: str = "mean",
    ) -> Float[Tensor, "B 1 H W"]:
        return ImgUtils.tensor2mask(ImgUtils.pil2tensor(img), mode=mode)

    @staticmethod
    def tensor2mask(
        x: Float[Tensor, "B C H W"],
        mode: MaskMode = "mean",
    ) -> Float[Tensor, "B 1 H W"]:
        """Reduce a multi-channel image tensor to a single-channel mask.

        Args:
            x: Image tensor (B, C, H, W).
            mode: Reduction mode. One of:
                - "R", "G", "B", "A": select the corresponding channel.
                - "RGB": mean of the first 3 channels (R, G, B). Useful when
                  an alpha channel was artificially added (e.g. ``=1``) and
                  should not dilute the color average.
                - "mean": mean across **all** channels.

        Returns:
            Single-channel mask (B, 1, H, W).

        Raises:
            ValueError: If mode is not one of "R", "G", "B", "A", "RGB", "mean".
        """
        channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
        if mode == "mean":
            return x.mean(dim=1, keepdim=True)
        if mode == "RGB":
            if x.shape[1] < 3:
                raise ValueError(
                    f"'RGB' mode requires at least 3 channels, got {x.shape[1]}."
                )
            return x[:, :3].mean(dim=1, keepdim=True)
        if mode in channel_map:
            idx = channel_map[mode]
            return x[:, idx : idx + 1, :, :]
        raise ValueError(
            f"Unknown mode '{mode}'; expected one of 'R', 'G', 'B', 'A', 'RGB', 'mean'."
        )

    @staticmethod
    def resize(
        img: Float[Tensor, "B C H W"],
        H: int,
        W: int,
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
        antialias: bool = False,
    ) -> Float[Tensor, "B C H W"]:
        """Resize image tensor to target height and width.

        Args:
            img: Input image (B, C, H_in, W_in).
            H: Target height.
            W: Target width.
            mode: Interpolation mode for F.interpolate (default "bilinear").
            align_corners: Optional align_corners argument.
            antialias: Apply antialiasing on downscaling.

        Returns:
            Resized image (B, C, H, W).

        Notes:
            - Requires batch dimension; never squeezes.
            - align_corners is ignored for "nearest" and "area" modes.
        """
        if H < 1 or W < 1:
            raise Exception(f"Target size must be positive, got H={H}, W={W}.")

        kwargs = {"antialias": antialias}
        if mode not in ("nearest", "area"):
            kwargs["align_corners"] = align_corners

        return F.interpolate(img, size=(H, W), mode=mode, **kwargs)

    @staticmethod
    @torch.no_grad()
    def ensure_rgba(img: Float[Tensor, "B C H W"]) -> Float[Tensor, "B 4 H W"]:
        """Ensure image has 4 channels (RGBA).

        Args:
            img: Input tensor (B, C, H, W).

        Returns:
            RGBA tensor (B, 4, H, W).
        """
        B, C, H, W = img.shape
        if C == 4:
            return img
        elif C == 3:
            return torch.cat(
                [img, torch.ones((B, 1, H, W), device=img.device, dtype=img.dtype)],
                dim=1,
            )
        elif C == 1:
            return torch.cat(
                [
                    img.repeat(1, 3, 1, 1),
                    torch.ones((B, 1, H, W), device=img.device, dtype=img.dtype),
                ],
                dim=1,
            )
        else:
            raise Exception(
                f"Cannot recognize image format for tensor with shape {img.shape}"
            )

    @staticmethod
    @torch.no_grad()
    def ensure_rgb(img: Float[Tensor, "B C H W"]) -> Float[Tensor, "B 4 H W"]:
        """Ensure image has 3 channels (RGB).

        Args:
            img: Input tensor (B, C, H, W).

        Returns:
            RGBA tensor (B, 4, H, W).
        """
        B, C, H, W = img.shape
        if C == 3:
            return img
        elif C == 4:
            return img[:, :3] * img[:, 3].unsqueeze(1)
        elif C == 1:
            return img.repeat(1, 3, 1, 1)
        else:
            raise Exception(
                f"Cannot recognize image format for tensor with shape {img.shape}"
            )

    @staticmethod
    @torch.no_grad()
    def gen_px_coords(
        H: int,
        W: int,
        device: torch.device,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Float[Tensor, "2 (H+pt+pb) (W+pl+pr)"]:
        """Generate normalized pixel coordinates.

        Both axes independently fill ``[0, 1]`` so each pixel center lies
        at ``(i + 0.5) / dim``. Padding is applied **outward** — the
        returned grid has shape ``(2, H + pt + pb, W + pl + pr)``.

        Args:
            H: Image height (unpadded frame).
            W: Image width (unpadded frame).
            device: Target device.
            padding: Outward padding (top, bottom, left, right).

        Returns:
            Coordinates tensor ``(2, H + pt + pb, W + pl + pr)``.
        """
        p_top, p_bot, p_left, p_right = padding
        y_start, y_end = 0.5 / H, 1.0 - 0.5 / H
        x_start, x_end = 0.5 / W, 1.0 - 0.5 / W

        co = torch.stack(
            torch.meshgrid(
                torch.linspace(y_start, y_end, H, device=device),
                torch.linspace(x_start, x_end, W, device=device),
                indexing="ij",
            ),
            dim=0,
        )
        return ImgUtils.coords_pad(co, padding=padding)

    @staticmethod
    def extract_patches(
        co: Float[Tensor, "C H W"], patch_size: Optional[int] = None
    ) -> Tuple[Float[Tensor, "P S C"], Float[Tensor, "P C"]]:
        """Extract patches from coordinate grid.

        Args:
            co: Coordinate tensor (C, H, W).
            patch_size: Size of patches to extract.

        Returns:
            Tuple of (patches, centers):
                - patches: (P, S, C) where P=num_patches, S=patch_size^2
                - centers: (P, C)
        """
        if patch_size < 1:
            raise Exception("Patch size must be strictly positive integer.")
        C, H, W = co.shape
        if patch_size is None or all(patch_size > d for d in [H, W]):
            patches = co.permute(1, 2, 0).reshape(1, H * W, C)  # [1, S, C]
            centers = torch.tensor(
                [0.5, 0.5], device=co.device, dtype=co.dtype
            ).unsqueeze(0)
            return patches, centers
        S = patch_size**2
        pad_H = (patch_size - (H % patch_size)) % patch_size
        pad_W = (patch_size - (W % patch_size)) % patch_size
        co = ImgUtils.coords_pad(co, padding=(0, pad_H, 0, pad_W))
        patches = (
            F.unfold(
                co.unsqueeze(0),
                kernel_size=patch_size,
                stride=patch_size,
            )
            .reshape(C, S, -1)
            .permute(2, 1, 0)
        )  # [P, S, C]
        centers = patches.mean(dim=1)
        return patches, centers

    @staticmethod
    def get_patches(
        H: int,
        W: int,
        device: torch.device,
        patch_size: Optional[int] = None,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Tuple[Float[Tensor, "P S 2"], Float[Tensor, "P 2"]]:
        """Get patches for image dimensions.

        Args:
            H: Image height.
            W: Image width.
            device: Target device.
            patch_size: Optional patch size.
            padding: Outward padding (top, bottom, left, right).

        Returns:
            Tuple of (patches, centers).
        """
        return ImgUtils.extract_patches(
            ImgUtils.gen_px_coords(H, W, device, padding=padding),
            patch_size,
        )

    @staticmethod
    def extract_image_patches(
        img: Float[Tensor, "B C H W"],
        patch_size: Optional[int],
        padding_mode: str = "replicate",
    ) -> Float[Tensor, "B P S C"]:
        """Extract image patches matching the layout of get_patches coordinates.

        Pads the image so H and W are divisible by patch_size, then extracts
        non-overlapping square patches using im2col/unfold.

        Args:
            img: Input image (B, C, H, W).
            patch_size: Size of square patches. If None or larger than both H and W,
                returns a single patch containing all H*W pixels.
            padding_mode: Padding mode for F.pad (default "replicate").

        Returns:
            Image patches (B, P, S, C). S is patch_size**2 normally, or H*W in
            the single-patch fallback.

        Notes:
            - Patch ordering matches ImgUtils.get_patches row-major layout.
            - Fallback behavior matches ImgUtils.extract_patches exactly.
            - Never squeezes batch dimension.
        """
        if patch_size is not None and patch_size < 1:
            raise Exception("Patch size must be strictly positive integer.")

        B, C, H, W = img.shape

        # Fallback: matches extract_patches when patch_size is None or > H and > W
        if patch_size is None or (patch_size > H and patch_size > W):
            return img.permute(0, 2, 3, 1).reshape(B, 1, H * W, C)  # (B, 1, H*W, C)

        S = patch_size * patch_size
        pad_H = (patch_size - (H % patch_size)) % patch_size
        pad_W = (patch_size - (W % patch_size)) % patch_size

        # F.pad order: (left, right, top, bottom) — pad only right/bottom
        padded = F.pad(
            img, (0, pad_W, 0, pad_H), mode=padding_mode
        )  # (B, C, H+pad_H, W+pad_W)

        patches = F.unfold(
            padded, kernel_size=patch_size, stride=patch_size
        )  # (B, C*S, P)
        patches = patches.view(B, C, S, -1).permute(0, 3, 2, 1)  # (B, P, S, C)
        return patches

    @staticmethod
    def coords_pad(
        co: Float[Tensor, "C H W"],
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Float[Tensor, "C (H+pad_top+pad_bottom) (W+pad_left+pad_right)"]:
        pad_top, pad_bottom, pad_left, pad_right = padding
        if padding == (0, 0, 0, 0):
            return co
        C, H, W = co.shape
        y_coords = co[0, :, 0]
        x_coords = co[1, 0, :]
        y_step = (
            y_coords[1] - y_coords[0]
            if H > 1
            else torch.tensor(1.0 / H, device=co.device, dtype=co.dtype)
        )
        x_step = (
            x_coords[1] - x_coords[0]
            if W > 1
            else torch.tensor(1.0 / W, device=co.device, dtype=co.dtype)
        )
        y_extra_top = (
            (
                y_coords[0]
                - torch.arange(1, pad_top + 1, device=co.device, dtype=co.dtype)
                * y_step
            ).flip(0)
            if pad_top > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        y_extra_bottom = (
            (
                y_coords[-1]
                + torch.arange(1, pad_bottom + 1, device=co.device, dtype=co.dtype)
                * y_step
            )
            if pad_bottom > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        x_extra_left = (
            (
                x_coords[0]
                - torch.arange(1, pad_left + 1, device=co.device, dtype=co.dtype)
                * x_step
            ).flip(0)
            if pad_left > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        x_extra_right = (
            (
                x_coords[-1]
                + torch.arange(1, pad_right + 1, device=co.device, dtype=co.dtype)
                * x_step
            )
            if pad_right > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        y_full = torch.cat([y_extra_top, y_coords, y_extra_bottom], dim=0)
        x_full = torch.cat([x_extra_left, x_coords, x_extra_right], dim=0)
        yy, xx = torch.meshgrid(y_full, x_full, indexing="ij")
        return torch.stack([yy, xx], dim=0)

    @staticmethod
    def assemble_patches(
        sampled_patches: Float[Tensor, "P S C"],
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> Float[Tensor, "B C H W"]:
        """Assemble sampled patches into full image.

        Args:
            sampled_patches: Sampled patches (P, S, C) where S=patch_size^2.
            H: Output height.
            W: Output width.

        Returns:
            Assembled image (B, C, H, W).
        """
        P, S, C = sampled_patches.shape
        patch_size = int(S**0.5)
        # Single-patch fallback (extract_image_patches returns S = H*W when
        # patch_size > H and > W; S is then not necessarily a perfect square,
        # so F.fold cannot reconstruct it). Reshape directly.
        if P == 1 and patch_size * patch_size != S:
            if H is None or W is None:
                raise Exception(
                    "assemble_patches needs H, W for the single-patch fallback."
                )
            if S != H * W:
                raise Exception(
                    f"single-patch fallback expects S == H*W, got S={S}, H*W={H * W}."
                )
            return sampled_patches.permute(2, 1, 0).reshape(1, C, H, W)  # (1, C, H, W)
        patches_H = math.ceil(H / patch_size)
        patches_W = math.ceil(W / patch_size)
        output_H = patch_size * patches_H
        output_W = patch_size * patches_W
        # if output_H * output_W != P:
        #     raise Exception(f"patches do not match image output size.")
        folded = F.fold(
            sampled_patches.permute(2, 1, 0).reshape(1, C * S, P),
            output_size=(output_H, output_W),
            kernel_size=patch_size,
            stride=patch_size,
        )
        return folded[:, :, :H, :W]

    @staticmethod
    @torch.no_grad()
    def gaussian_kernel(
        kernel_size: int | Sequence[int],
        sigma: float | Sequence[float],
    ) -> Float[Tensor, "1 1 KH KW"]:
        """Generate 2D Gaussian kernel.

        Args:
            kernel_size: Kernel size (int or [H, W]).
            sigma: Standard deviation (float or [H, W]).

        Returns:
            Gaussian kernel (1, 1, KH, KW).
        """
        if isinstance(kernel_size, int):
            kH, kW = kernel_size, kernel_size
        elif len(kernel_size) == 1:
            kH, kW = kernel_size[0], kernel_size[0]
        elif len(kernel_size) == 2:
            kH, kW = kernel_size
        else:
            raise Exception(
                f"2D gaussian kernel cannot accept more than 2 axes. Kernel '{kernel_size} is too long."
            )
        if isinstance(sigma, int):
            sH, sW = sigma, sigma
        elif len(sigma) == 1:
            sH, sW = sigma[0], sigma[0]
        elif len(sigma) == 2:
            sH, sW = sigma
        else:
            raise Exception(
                f"2D gaussian kernel cannot accept more than 2 axes. Sigmas '{kernel_size} are too many."
            )
        sq2pi = (2 * math.pi) ** 0.5
        muH = (kH - 1) / 2
        xH = torch.exp(-0.5 * ((torch.linspace(0, kH - 1, kH) - muH) / sH) ** 2) / (
            sH * sq2pi
        )
        if kH == kW and sH == sW:
            xW = xH
        else:
            muW = (kW - 1) / 2
            xW = torch.exp(-0.5 * ((torch.linspace(0, kW - 1, kW) - muW) / sW) ** 2) / (
                sW * sq2pi
            )
        return (xH.unsqueeze(1) @ xW.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def convolve(
        img: Float[Tensor, "B C H W"],
        kernel: Float[Tensor, "C G KH KW"],
        match_channels: bool = False,
        stride: int | Tuple[int] = 1,
        padding: str = "same",
    ) -> Float[Tensor, "B C H W"]:
        """Convolve image with kernel.

        Args:
            img: Input image (B, C, H, W).
            kernel: Convolution kernel (C, G, KH, KW).
            match_channels: If True, expand kernel to match input channels.
            stride: Convolution stride.
            padding: Padding mode ("same" or "valid").

        Returns:
            Convolved output (B, C, H, W).
        """
        Bi, Ci, Hi, Wi = img.shape
        Ck, Gk, Hk, Wk = kernel.shape
        if match_channels:
            if Ck != 1 or Gk != 1:
                raise Exception(
                    f"Cannot match channels for kernel with non-singleton dimensions.\nKernel shape: {kernel.shape}"
                )
            kernel = kernel.expand(Ci, Ci, -1, -1)
        return F.conv2d(img, kernel.to(img.device), stride=stride, padding=padding)

    @staticmethod
    def SSIM(
        img1: Float[Tensor, "B C H W"],
        img2: Float[Tensor, "B C H W"],
        kernel: Float[Tensor, "1 1 KH KW"],
        eps1: float = 0.0004,
        eps2: float = 0.0036,
    ) -> Float[Tensor, "B C H W"]:
        """Compute Structural Similarity Index (SSIM).

        Args:
            img1: First image (B, C, H, W).
            img2: Second image (B, C, H, W).
            kernel: Gaussian kernel (1, 1, KH, KW).
            eps1: Stability constant for means.
            eps2: Stability constant for variances.

        Returns:
            SSIM map (B, C, H, W) with values in [-1, 1].
        """
        mux = ImgUtils.convolve(img1, kernel, match_channels=True)
        muy = ImgUtils.convolve(img2, kernel, match_channels=True)
        mu2x = mux**2
        mu2y = muy**2
        sig2x = ImgUtils.convolve(img1**2, kernel, match_channels=True) ** 2 - mu2x
        sig2y = ImgUtils.convolve(img2**2, kernel, match_channels=True) ** 2 - mu2y
        sigxy = (
            ImgUtils.convolve(img1 * img2, kernel, match_channels=True) ** 2 - mux * muy
        )
        return (
            (2 * mux * muy + eps1)
            * (sigxy + eps2)
            / ((mu2x + mu2y + eps2) * (sig2x + sig2y + eps2))
        )

    @staticmethod
    def load_image(path: str, mode: str = "RGBA") -> Float[Tensor, "B C H W"]:
        """Load image from path as tensor.

        Args:
            path: Path to image file (PNG, JPG, etc.).
            mode: Color mode for PIL Image ("RGBA", "RGB", "L", etc.).

        Returns:
            Image tensor (B, C, H, W) with values in [0, 1].
        """
        img = Image.open(path).convert(mode)
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).permute(0, 3, 1, 2)

    @staticmethod
    def img2mask(
        img: Float[Tensor, "B C H W"], min: float = -1.0, max: float = 1.0
    ) -> Float[Tensor, "B 1 H W"]:
        return (img.mean(dim=1).unsqueeze(1) + 1) / 2

    @staticmethod
    def load_mask(path: str) -> Float[Tensor, "B 1 H W"]:
        return ImgUtils.img2mask(ImgUtils.load_image(path, mode="RGBA"))

    @staticmethod
    def same_size(*imgs: Float[Tensor, "B C H W"]) -> bool:
        if len(imgs) <= 1:
            raise ValueError(
                f"Function requires a minimum of 2 images to compare. Provided {len(imgs)}."
            )
        ref = imgs[0].shape[-2:]
        for i in imgs[1:]:
            if i.shape[-2:] != ref:
                return False
        return True

    @staticmethod
    def uv_sample(
        img: Float[Tensor, "B C H W"],
        uv_co: Float[Tensor, "N 2"],
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Float[Tensor, "B N C"]:
        """Bilinearly sample an image at normalized pixel-center coordinates.

        Each axis independently fills ``[0, 1]``, and the pixel index for
        UV ``u`` is ``u * D - 0.5`` where ``D`` is the axis size.

        UVs outside ``[0, 1]`` (e.g. centroids that have drifted past the
        image frame after a split) are clamped to the nearest edge pixel
        via the fractional weights, so they return a valid sample rather
        than extrapolating.

        Implicit padding is treated as already baked into the image:
        ``padding`` (top, bottom, left, right) shifts UV ``[0, 1]`` to the
        logical (unpadded) frame so coordinates falling inside the padded
        border correctly sample those border pixels.

        Args:
            img: Image tensor (B, C, H, W).
            uv_co: Normalized coordinates (N, 2) in pixel-center convention,
                typically produced by :meth:`gen_px_coords` or by
                ``primitive.centroid_coordinates``.
            padding: Implicit padding (top, bottom, left, right) already
                present in ``img``. UV ``[0, 1]`` maps to the logical
                frame; values outside that range sample the padding.

        Returns:
            Sampled values (B, N, C).
        """
        B, C, H, W = img.shape
        N = uv_co.shape[0]

        pt, pb, pl, pr = padding
        H_logical = H - pt - pb
        W_logical = W - pl - pr
        if H_logical < 1 or W_logical < 1:
            raise ValueError(
                f"Padding {padding} leaves no logical image (got "
                f"H_logical={H_logical}, W_logical={W_logical} from "
                f"img shape {(B, C, H, W)})."
            )

        y = uv_co[:, 0] * H_logical - 0.5 + pt
        x = uv_co[:, 1] * W_logical - 0.5 + pl

        x0 = x.floor().long().clamp(0, W - 1)
        y0 = y.floor().long().clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        # Clamp fractional weights so out-of-frame UVs clamp cleanly to the
        # nearest edge pixel instead of extrapolating with negative or >1
        # weights.
        fx = (x - x0.float()).clamp(0, 1).view(1, 1, N)
        fy = (y - y0.float()).clamp(0, 1).view(1, 1, N)
        inv_fx = 1 - fx
        inv_fy = 1 - fy

        tl = img[:, :, y0, x0]
        tr = img[:, :, y0, x1]
        bl = img[:, :, y1, x0]
        br = img[:, :, y1, x1]

        vals = (
            tl * (inv_fx * inv_fy)
            + tr * (fx * inv_fy)
            + bl * (inv_fx * fy)
            + br * (fx * fy)
        )
        return vals.permute(0, 2, 1)  # (B, N, C)

    @staticmethod
    @torch.no_grad()
    def sample_px_coords(
        map: Float[Tensor, "B 1 H W"],
        N: int,
        noise: bool = False,
    ) -> Float[Tensor, "N 2"]:
        """Sample N pixel coordinates weighted by a map.

        Args:
            map: Weight map (B, 1, H, W) with non-negative values.
            N: Number of coordinates to sample.
            noise: If True, jitter each sampled coordinate uniformly within
                half a pixel on either side of its pixel center.

        Returns:
            Sampled coordinates (N, 2) in pixel-center convention.

        Notes:
            - Sampling uses ``torch.multinomial`` with replacement, so
              coordinates may repeat.
            - Coordinates follow the pixel-center convention of
              :meth:`gen_px_coords` (centered at ``(i + 0.5) / D``).
            - With ``noise=True``, jitter is uniform in ``[-0.5/H, 0.5/H]`` for
              y and ``[-0.5/W, 0.5/W]`` for x, i.e. exactly one pixel width
              centered on each pixel.
        """
        B, _, H, W = map.shape
        if B != 1:
            raise ValueError(
                f"sample_px_coords expects a single-map batch (B=1), got B={B}."
            )
        weights = map.reshape(-1)  # (H*W,)
        if (weights < 0).any():
            raise ValueError("sample_px_coords expects non-negative map values.")
        indices = torch.multinomial(weights, N, replacement=True)  # (N,)
        co = ImgUtils.gen_px_coords(H, W, map.device)  # (2, H, W)
        co_flat = co.reshape(2, -1).T  # (H*W, 2)
        sampled = co_flat[indices]  # (N, 2)
        if noise:
            sampled = sampled.clone()
            sampled[:, 0].add_((torch.rand(N, device=map.device) - 0.5) / H)
            sampled[:, 1].add_((torch.rand(N, device=map.device) - 0.5) / W)
        return sampled

    @staticmethod
    @torch.no_grad()
    def distance_map(
        coords: Float[Tensor, "Nc 2"],
        H: int,
        W: int,
        mode: str = "MIN",
        k: int = 1,
    ) -> Float[Tensor, "1 1 H W"]:
        """Compute a per-pixel distance map to a set of 2D coordinates.

        Pixel centers follow :meth:`gen_px_coords`; distances are Euclidean.

        Args:
            coords: Coordinates (Nc, 2) in pixel-center convention
                (values in [0, 1]).
            H: Output map height.
            W: Output map width.
            mode: Reduction mode. One of:
                - ``"MIN"``: distance to the closest point.
                - ``"MIN_K"``: mean distance to the ``k`` closest points.
                - ``"MEAN"``: mean distance to all points.
                - ``"MAX"``: distance to the farthest point.
                - ``"MAX_K"``: mean distance to the ``k`` farthest points.
            k: Number of points for ``MIN_K`` / ``MAX_K``; ignored by other
                modes.

        Returns:
            Distance map (1, 1, H, W).

        Raises:
            ValueError: If ``mode`` is unknown, ``Nc == 0``, or ``k`` is not
                in ``[1, Nc]``.

        Notes:
            - Distances are computed in normalized pixel-center space, so
              values lie in ``[0, sqrt(2)]``.
        """
        Nc = coords.shape[0]
        if Nc == 0:
            raise ValueError("distance_map requires at least one coordinate.")
        if k < 1 or k > Nc:
            raise ValueError(f"k must be in [1, {Nc}], got k={k}.")

        co_grid = ImgUtils.gen_px_coords(H, W, coords.device)  # (2, H, W)
        co_flat = co_grid.reshape(2, -1).T  # (H*W, 2)
        dists = torch.cdist(co_flat, coords.to(co_flat.dtype))  # (H*W, Nc)

        if mode == "MIN":
            reduced = dists.min(dim=1).values
        elif mode == "MIN_K":
            reduced = torch.topk(dists, k, largest=False).values.mean(dim=1)
        elif mode == "MEAN":
            reduced = dists.mean(dim=1)
        elif mode == "MAX":
            reduced = dists.max(dim=1).values
        elif mode == "MAX_K":
            reduced = torch.topk(dists, k, largest=True).values.mean(dim=1)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'; expected one of "
                f"'MIN', 'MIN_K', 'MEAN', 'MAX', 'MAX_K'."
            )

        return reduced.reshape(1, 1, H, W)

    @staticmethod
    @torch.no_grad()
    def delta_map(
        coords: Float[Tensor, "Nc 2"],
        H: int,
        W: int,
        mode: str = "MIN",
        k: int = 1,
    ) -> Tuple[Float[Tensor, "1 1 H W"], Float[Tensor, "1 1 H W"]]:
        """Compute per-axis delta maps to a set of 2D coordinates.

        Same reductions as :meth:`distance_map` but applied independently
        along the y and x axes using **signed** coordinate differences
        (``pixel_axis - coord_axis``) along each axis.

        Args:
            coords: Coordinates (Nc, 2) in pixel-center convention
                (values in [0, 1]). First column is y, second is x.
            H: Output map height.
            W: Output map width.
            mode: Reduction mode. One of:
                - ``"MIN"``: smallest signed delta (per axis).
                - ``"MIN_K"``: mean of the ``k`` smallest signed deltas (per axis).
                - ``"MEAN"``: mean signed delta (per axis).
                - ``"MAX"``: largest signed delta (per axis).
                - ``"MAX_K"``: mean of the ``k`` largest signed deltas (per axis).
            k: Number of points for ``MIN_K`` / ``MAX_K``; ignored by other
                modes.

        Returns:
            Tuple of (y_delta, x_delta). Each map (1, 1, H, W) holds per-axis
            signed delta values in normalized units.

        Raises:
            ValueError: If ``mode`` is unknown, ``Nc == 0``, or ``k`` is not
                in ``[1, Nc]``.
        """
        Nc = coords.shape[0]
        if Nc == 0:
            raise ValueError("delta_map requires at least one coordinate.")
        if k < 1 or k > Nc:
            raise ValueError(f"k must be in [1, {Nc}], got k={k}.")

        co_grid = ImgUtils.gen_px_coords(H, W, coords.device)  # (2, H, W)
        y_axis = co_grid[0].reshape(-1).to(coords.dtype)  # (H*W,)
        x_axis = co_grid[1].reshape(-1).to(coords.dtype)  # (H*W,)

        y_diff = y_axis[:, None] - coords[:, 0]  # (H*W, Nc)
        x_diff = x_axis[:, None] - coords[:, 1]  # (H*W, Nc)

        def _reduce(
            diffs: Float[Tensor, "HW Nc"],
        ) -> Float[Tensor, " HW"]:
            if mode == "MIN":
                return diffs.min(dim=1).values
            if mode == "MIN_K":
                return torch.topk(diffs, k, largest=False).values.mean(dim=1)
            if mode == "MEAN":
                return diffs.mean(dim=1)
            if mode == "MAX":
                return diffs.max(dim=1).values
            if mode == "MAX_K":
                return torch.topk(diffs, k, largest=True).values.mean(dim=1)
            raise ValueError(
                f"Unknown mode '{mode}'; expected one of "
                f"'MIN', 'MIN_K', 'MEAN', 'MAX', 'MAX_K'."
            )

        y_delta = _reduce(y_diff).reshape(1, 1, H, W)
        x_delta = _reduce(x_diff).reshape(1, 1, H, W)
        return y_delta, x_delta


_SplimageInput = Union[
    np.ndarray,
    Image.Image,
    List[Image.Image],
    Tensor,
    os.PathLike,
    str,
]


class Splimage:
    """Cached image wrapper with lazy format conversion and padding.

    Canonical representation is a BCHW torch tensor in [0, 1].
    Conversions to numpy, PIL, and mask are cached on first access and
    invalidated by any mutation.

    Attributes:
        padding (Tuple[int, int, int, int]): Logical padding (top, bottom,
            left, right) in pixels. Used by uv_sample to shift coordinates
            inward.

    Construction:
        Splimage(img): Construct from numpy (BHWC), PIL Image or list of
            PIL Images, or torch tensor (BCHW).
    """

    _tensor: Float[Tensor, "B C H W"]
    _padding: Tuple[int, int, int, int]
    _img_cache: Optional[Float[Tensor, "B C H W"]]
    _pil_cache: Optional[Union[Image.Image, List[Image.Image]]]
    _np_cache: Optional[np.ndarray]
    _mask_cache: dict
    _forced_alpha: bool
    _force_rgba_enabled: bool
    _mask_mode: MaskMode
    _is_mask: bool
    _as_mask_enabled: bool

    def __init__(
        self,
        img: _SplimageInput,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        force_rgba: bool = True,
        mask_mode: MaskMode = "mean",
        as_mask: bool = False,
    ) -> None:
        """
        Args:
            img: Source image. Accepted formats:
                - numpy array: BHWC (or HWC, auto-batched).
                - PIL Image or list of PIL Images.
                - torch tensor: BCHW (or HWC, auto-batched).
                - path-like (``str`` or ``os.PathLike``): loaded via
                  :meth:`PIL.Image.open` and converted to a tensor.
            padding: Implicit padding (top, bottom, left, right) already
                present in the image; not added by this constructor.
                Capped by the image's H/W.
            force_rgba: If True, pad multi-channel inputs with fewer than
                4 channels with a constant alpha of ``1`` so the canonical
                tensor is BCHW with C=4. Inputs that arrive as 1-channel
                tensors are treated as masks and kept as-is regardless of
                this flag; ``mask(mode="mean")`` on a forced-alpha image is
                automatically routed through the first 3 channels (RGB) so
                the artificial alpha does not dilute the average.
            mask_mode: Mode used by :meth:`mask` when called
                without an explicit ``mode`` argument. Persists across
                :meth:`update` calls.
            as_mask: If True, the input is reduced to a single-channel
                mask via ``mask_mode`` and stored as the canonical tensor.
                The resulting Splimage behaves like a mask-initialized one:
                ``_is_mask=True``, all tensor ops target the mask, and
                ``mask()`` returns it directly. ``force_rgba`` is bypassed
                in this case. No-op for inputs that are already 1-channel.
        """
        self._img_cache = None
        self._pil_cache = None
        self._np_cache = None
        self._mask_cache = {}
        self._forced_alpha = False
        self._is_mask = False
        self._force_rgba_enabled = force_rgba
        self._mask_mode = mask_mode
        self._as_mask_enabled = as_mask

        if isinstance(img, np.ndarray):
            self._from_numpy(img)
        elif isinstance(img, Image.Image):
            self._tensor = ImgUtils.pil2tensor(img)
        elif isinstance(img, (list, tuple)):
            self._tensor = ImgUtils.pil2tensor(img)
        elif isinstance(img, Tensor):
            self._from_tensor(img)
        elif isinstance(img, (str, os.PathLike)):
            self._tensor = ImgUtils.pil2tensor(Image.open(img))
        else:
            raise TypeError(f"Cannot construct Splimage from {type(img).__name__}.")

        if as_mask and self._tensor.shape[1] != 1:
            self._tensor = ImgUtils.tensor2mask(self._tensor, mode=mask_mode)
            self._is_mask = True
        elif self._tensor.shape[1] == 1:
            self._is_mask = True
        elif force_rgba and self._tensor.shape[1] != 4:
            self._force_alpha()

        self.set_padding(padding)

    def _force_alpha(self) -> None:
        B, C, H, W = self._tensor.shape
        if C == 4:
            self._forced_alpha = False
            return
        alpha = torch.ones(
            (B, 1, H, W),
            device=self._tensor.device,
            dtype=self._tensor.dtype,
        )
        if C == 1:
            rgb = self._tensor.repeat(1, 3, 1, 1)
            self._tensor = torch.cat([rgb, alpha], dim=1)
        elif C == 3:
            self._tensor = torch.cat([self._tensor, alpha], dim=1)
        else:
            raise ValueError(f"Cannot force RGBA from {C} channels.")
        self._forced_alpha = True

    def _from_numpy(self, arr: np.ndarray) -> None:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3:
            arr = arr[np.newaxis]
        if arr.ndim != 4:
            raise ValueError(f"Expected 2D-4D numpy array, got shape {arr.shape}.")
        C = arr.shape[3]
        if C not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {C}.")
        self._tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    def _from_tensor(self, t: Tensor) -> None:
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0).expand(3, -1, -1, -1)
        if t.ndim == 3:
            C = t.shape[0]
            if C in (1, 3, 4):
                t = t.unsqueeze(0)
            elif t.shape[2] in (1, 3, 4):
                t = t.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"Ambiguous 3D tensor shape {t.shape}; "
                    f"expected CHW or HWC with C in (1, 3, 4)."
                )
        if t.ndim != 4:
            raise ValueError(f"Expected 2D-4D tensor, got shape {t.shape}.")
        C = t.shape[1]
        if C not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {C}.")
        self._tensor = t.float()

    def _invalidate(self) -> None:
        self._img_cache = None
        self._pil_cache = None
        self._np_cache = None
        self._mask_cache.clear()

    def copy(self) -> Splimage:
        """Return a deep copy of this Splimage.

        The underlying tensor is cloned, padding/mode metadata is
        duplicated, and caches start empty.
        """
        new = Splimage.__new__(Splimage)
        new._tensor = self._tensor.clone()
        new._padding = self._padding
        new._forced_alpha = self._forced_alpha
        new._is_mask = self._is_mask
        new._force_rgba_enabled = self._force_rgba_enabled
        new._mask_mode = self._mask_mode
        new._as_mask_enabled = self._as_mask_enabled
        new._img_cache = None
        new._pil_cache = None
        new._np_cache = None
        new._mask_cache = {}
        return new

    def update(
        self,
        img: _SplimageInput,
        padding: Optional[Tuple[int, int, int, int]] = None,
        force_rgba: Optional[bool] = None,
        mask_mode: Optional[MaskMode] = None,
        as_mask: Optional[bool] = None,
    ) -> None:
        """Replace the underlying image and invalidate all caches.

        Args:
            img: New image (numpy BHWC, PIL Image, torch BCHW tensor, or path-like).
            padding: Optional implicit padding (top, bottom, left, right)
                for the new image. If None, the current padding is kept.
            force_rgba: If True, pad to 4 channels with alpha=1. If False,
                leave the channel count as-is. If None, use the setting from
                construction.
            mask_mode: Optional override for the default mask
                mode used by :meth:`mask`. If None, the existing setting
                is kept.
            as_mask: If True, reduce the new input to a 1-channel mask via
                ``mask_mode`` and store as the canonical tensor. If None,
                use the setting from construction.
        """
        if isinstance(img, np.ndarray):
            self._from_numpy(img)
        elif isinstance(img, Image.Image):
            self._tensor = ImgUtils.pil2tensor(img)
        elif isinstance(img, (list, tuple)):
            self._tensor = ImgUtils.pil2tensor(img)
        elif isinstance(img, Tensor):
            self._from_tensor(img)
        elif isinstance(img, (str, os.PathLike)):
            self._tensor = ImgUtils.pil2tensor(Image.open(img))
        else:
            raise TypeError(f"Cannot update Splimage from {type(img).__name__}.")

        self._forced_alpha = False
        if as_mask is None:
            as_mask = self._as_mask_enabled
        if as_mask and self._tensor.shape[1] != 1:
            self._tensor = ImgUtils.tensor2mask(self._tensor, mode=self._mask_mode)
            self._is_mask = True
        else:
            self._is_mask = self._tensor.shape[1] == 1
            if not self._is_mask:
                if force_rgba is None:
                    force_rgba = self._force_rgba_enabled
                if force_rgba and self._tensor.shape[1] != 4:
                    self._force_alpha()

        self._invalidate()
        if padding is not None:
            self.set_padding(padding)
        if mask_mode is not None:
            self._mask_mode = mask_mode

    @property
    def padding(self) -> Tuple[int, int, int, int]:
        return self._padding

    @padding.setter
    def padding(self, value: Tuple[int, int, int, int]) -> None:
        self.set_padding(value)

    def set_padding(self, value: Tuple[int, int, int, int]) -> None:
        """Set implicit padding (top, bottom, left, right).

        The image is **not** padded; this only records metadata so
        :meth:`uv_sample` can correctly interpret coordinates that fall
        inside the border region.

        Args:
            value: Padding (top, bottom, left, right). Each entry is
                clamped to ``[0, dim - 1]`` where ``dim`` is the
                corresponding axis size.

        Raises:
            ValueError: If ``value`` does not have length 4 or has negative
                entries.
        """
        if len(value) != 4:
            raise ValueError(f"Padding must have length 4, got {value}.")
        if any(v < 0 for v in value):
            raise ValueError(f"Padding entries must be non-negative, got {value}.")
        pt, pb, pl, pr = value
        pt = min(pt, max(self.H - 1, 0))
        pb = min(pb, max(self.H - pt - 1, 0))
        pl = min(pl, max(self.W - 1, 0))
        pr = min(pr, max(self.W - pl - 1, 0))
        self._padding = (pt, pb, pl, pr)

    def _hw_from_cache(self) -> Optional[Tuple[int, int]]:
        """Return (H, W) from the first populated cache, else None."""
        if self._img_cache is not None:
            t = self._img_cache
            return t.shape[2], t.shape[3]
        if self._mask_cache:
            t = next(iter(self._mask_cache.values()))
            return t.shape[2], t.shape[3]
        if self._pil_cache is not None:
            pil = self._pil_cache
            first = pil[0] if isinstance(pil, list) else pil
            return first.size[1], first.size[0]
        if self._np_cache is not None:
            arr = self._np_cache
            return arr.shape[1], arr.shape[2]
        return None

    @property
    def H(self) -> int:
        hw = self._hw_from_cache()
        if hw is not None:
            return hw[0]
        return self._tensor.shape[2]

    @property
    def W(self) -> int:
        hw = self._hw_from_cache()
        if hw is not None:
            return hw[1]
        return self._tensor.shape[3]

    @property
    def C(self) -> int:
        return self._tensor.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return tuple(self._tensor.shape)

    def same_size(self, *others: Splimage) -> bool:
        """Check whether this image has the same H and W as the others.

        Uses whatever representation is already cached on each Splimage
        (avoiding any new tensor conversion) and falls back to the
        underlying tensor when no cache is populated.

        Args:
            others: Other Splimage instances.

        Returns:
            True if all images have matching H and W.

        Raises:
            ValueError: If fewer than two images are provided.
        """
        images = (self, *others)
        if len(images) < 2:
            raise ValueError(
                f"same_size requires a minimum of 2 images to compare, "
                f"got {len(images)}."
            )
        ref = (images[0].H, images[0].W)
        for img in images[1:]:
            if (img.H, img.W) != ref:
                return False
        return True

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def to(self, *args, **kwargs) -> Splimage:
        """Move/cast the underlying tensor. Returns a new Splimage.

        Forwards all keyword arguments to ``Tensor.to`` (e.g. ``device``,
        ``dtype``, ``non_blocking``, ``memory_format``).

        Returns:
            New Splimage with the converted tensor. Padding and
            ``_forced_alpha`` are preserved.
        """
        new = Splimage.__new__(Splimage)
        new._tensor = self._tensor.to(*args, **kwargs)
        new._padding = self._padding
        new._forced_alpha = self._forced_alpha
        new._is_mask = self._is_mask
        new._force_rgba_enabled = self._force_rgba_enabled
        new._mask_mode = self._mask_mode
        new._as_mask_enabled = self._as_mask_enabled
        new._img_cache = None
        new._pil_cache = None
        new._np_cache = None
        new._mask_cache = {}
        return new

    def image(self) -> Float[Tensor, "B C H W"]:
        """Return BCHW tensor. Cached after first call.

        For mask-initialized Splimages, returns the canonical 1-channel
        tensor directly (the mask **is** the image).
        """
        if self._img_cache is None:
            self._img_cache = self._tensor
        return self._img_cache

    def mask(self, mode: Optional[MaskMode] = None) -> Float[Tensor, "B 1 H W"]:
        """Return single-channel mask. Cached per mode.

        Args:
            mode: Reduction mode ("R", "G", "B", "A", "RGB", or "mean").
                If None, the ``mask_mode`` set at construction is
                used. When the image's alpha was forced by
                ``force_rgba=True`` and the resolved mode is "mean", the
                call is automatically routed through "RGB" so the
                artificial alpha does not dilute the average. For
                mask-initialized Splimages, the canonical 1-channel
                tensor is returned directly regardless of ``mode``.
        """
        if self._is_mask:
            return self._tensor
        if mode is None:
            mode = self._mask_mode
        if mode == "mean" and self._forced_alpha:
            mode = "RGB"
        if mode not in self._mask_cache:
            self._mask_cache[mode] = ImgUtils.tensor2mask(self._tensor, mode=mode)
        return self._mask_cache[mode]

    def to_pil(self) -> Union[Image.Image, List[Image.Image]]:
        """Convert to PIL Image(s). Cached after first call."""
        if self._pil_cache is None:
            self._pil_cache = ImgUtils.tensor2pil(self._tensor)
        return self._pil_cache

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy BHWC uint8 array. Cached after first call."""
        if self._np_cache is None:
            img = ImgUtils.tensor2img(self._tensor, clamp=True)
            self._np_cache = (img.cpu().numpy() * 255).astype(np.uint8)
        return self._np_cache

    def show(self) -> None:
        """Display the image via PIL's default viewer.

        For batched images (B > 1), opens each frame in sequence. 1-channel
        tensors are broadcast to RGB for display. The viewer is
        platform-dependent; in headless environments this may raise an error.
        """
        if self._tensor.shape[1] == 1:
            rgb = self._tensor.repeat(1, 3, 1, 1)
            pil = ImgUtils.tensor2pil(rgb)
        else:
            pil = self.to_pil()
        if isinstance(pil, list):
            for img in pil:
                img.show()
        else:
            pil.show()

    def __add__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, torch.add)

    def __radd__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, lambda a, b: torch.add(b, a))

    def __iadd__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        self._tensor = torch.add(self._tensor, self._resolve_tensor(other))
        self._invalidate()
        return self

    def __sub__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, torch.sub)

    def __rsub__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, lambda a, b: torch.sub(b, a))

    def __isub__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        self._tensor = torch.sub(self._tensor, self._resolve_tensor(other))
        self._invalidate()
        return self

    def __mul__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, torch.mul)

    def __rmul__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, lambda a, b: torch.mul(b, a))

    def __imul__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        self._tensor = torch.mul(self._tensor, self._resolve_tensor(other))
        self._invalidate()
        return self

    def __truediv__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, torch.div)

    def __rtruediv__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        return self._apply_op(other, lambda a, b: torch.div(b, a))

    def __itruediv__(self, other: Union[Splimage, float, int, Tensor]) -> Splimage:
        self._tensor = torch.div(self._tensor, self._resolve_tensor(other))
        self._invalidate()
        return self

    def _resolve_tensor(self, other: Union[Splimage, float, int, Tensor]) -> Tensor:
        if isinstance(other, Splimage):
            return other._tensor
        return other

    def _apply_op(
        self, other: Union[Splimage, float, int, Tensor], op: Callable
    ) -> Splimage:
        result = op(self._tensor, self._resolve_tensor(other))
        new = Splimage(result)
        new._padding = self._padding
        return new

    def cos(self) -> Splimage:
        """Element-wise cosine. Returns new Splimage."""
        return self._unary(torch.cos)

    def cos_(self) -> Splimage:
        """Element-wise cosine in-place."""
        return self._unary_(torch.cos)

    def abs(self) -> Splimage:
        """Element-wise absolute value. Returns new Splimage."""
        return self._unary(torch.abs)

    def abs_(self) -> Splimage:
        """Element-wise absolute value in-place."""
        return self._unary_(torch.abs)

    def clamp(self, min: float = 0.0, max: float = 1.0) -> Splimage:
        """Clamp values. Returns new Splimage."""
        new = Splimage(torch.clamp(self._tensor, min, max))
        new._padding = self._padding
        return new

    def clamp_(self, min: float = 0.0, max: float = 1.0) -> Splimage:
        """Clamp values in-place."""
        self._tensor = torch.clamp(self._tensor, min, max)
        self._invalidate()
        return self

    def exp(self) -> Splimage:
        """Element-wise exponential. Returns new Splimage."""
        return self._unary(torch.exp)

    def exp_(self) -> Splimage:
        """Element-wise exponential in-place."""
        return self._unary_(torch.exp)

    def _unary(self, fn: Callable) -> Splimage:
        new = Splimage(fn(self._tensor))
        new._padding = self._padding
        return new

    def _unary_(self, fn: Callable) -> Splimage:
        self._tensor = fn(self._tensor)
        self._invalidate()
        return self

    def blur(self, kernel_size: int, sigma: Optional[float] = None) -> Splimage:
        """Apply a Gaussian blur via convolution. Returns new Splimage.

        Args:
            kernel_size: Kernel side length. Must be a positive odd integer.
            sigma: Standard deviation for the Gaussian kernel. Defaults to
                ``0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`` (OpenCV
                heuristic), which falls back to ``kernel_size / 6`` for
                ``kernel_size >= 3``.

        Returns:
            New Splimage with blurred tensor. Padding metadata is preserved.
        """
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = ImgUtils.gaussian_kernel([kernel_size], [sigma])
        blurred = ImgUtils.convolve(self._tensor, kernel, match_channels=True)
        new = Splimage(blurred, padding=self._padding)
        return new

    def blur_(self, kernel_size: int, sigma: Optional[float] = None) -> Splimage:
        """Apply a Gaussian blur in-place.

        Args:
            kernel_size: Kernel side length. Must be a positive odd integer.
            sigma: Standard deviation for the Gaussian kernel. Defaults to
                ``0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`` (OpenCV
                heuristic).
        """
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = ImgUtils.gaussian_kernel([kernel_size], [sigma])
        self._tensor = ImgUtils.convolve(self._tensor, kernel, match_channels=True)
        self._invalidate()
        return self

    def resize(
        self,
        H: int,
        W: int,
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
        antialias: bool = False,
    ) -> Splimage:
        """Resize image. Returns new Splimage.

        Args:
            H: Target height.
            W: Target width.
            mode: Interpolation mode.
            align_corners: Optional align_corners for F.interpolate.
            antialias: Apply antialiasing on downscaling.
        """
        new = Splimage(
            ImgUtils.resize(
                self._tensor,
                H,
                W,
                mode=mode,
                align_corners=align_corners,
                antialias=antialias,
            )
        )
        new._padding = self._padding
        return new

    def resize_(
        self,
        H: int,
        W: int,
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
        antialias: bool = False,
    ) -> Splimage:
        """Resize image in-place.

        Args:
            H: Target height.
            W: Target width.
            mode: Interpolation mode.
            align_corners: Optional align_corners for F.interpolate.
            antialias: Apply antialiasing on downscaling.
        """
        self._tensor = ImgUtils.resize(
            self._tensor,
            H,
            W,
            mode=mode,
            align_corners=align_corners,
            antialias=antialias,
        )
        self._invalidate()
        return self

    def convolve(
        self,
        kernel: Float[Tensor, "C G KH KW"],
        match_channels: bool = False,
        stride: int = 1,
        padding: str = "same",
    ) -> Splimage:
        """Convolve with kernel. Returns new Splimage.

        Args:
            kernel: Convolution kernel.
            match_channels: Expand kernel to match input channels.
            stride: Convolution stride.
            padding: Padding mode ("same" or "valid").
        """
        new = Splimage(
            ImgUtils.convolve(
                self._tensor,
                kernel,
                match_channels=match_channels,
                stride=stride,
                padding=padding,
            )
        )
        new._padding = self._padding
        return new

    def SSIM(
        self,
        other: Splimage,
        kernel: Float[Tensor, "1 1 KH KW"],
        eps1: float = 0.0004,
        eps2: float = 0.0036,
    ) -> Float[Tensor, "B C H W"]:
        """Compute SSIM between this and another Splimage.

        Args:
            other: Other image.
            kernel: Gaussian kernel.
            eps1: Stability constant for means.
            eps2: Stability constant for variances.

        Returns:
            SSIM map (B, C, H, W).
        """
        return ImgUtils.SSIM(self._tensor, other._tensor, kernel, eps1=eps1, eps2=eps2)

    def image_sample(
        self,
        uv_co: Float[Tensor, "N 2"],
    ) -> Float[Tensor, "B N C"]:
        """Bilinearly sample the full image at normalized coordinates.

        UV ``[0, 1]`` maps to the logical (unpadded) frame; values outside
        that range sample the implicit padding region.

        Args:
            uv_co: Normalized coordinates (N, 2) in pixel-center convention.

        Returns:
            Sampled values (B, N, C).
        """
        return ImgUtils.uv_sample(self._tensor, uv_co, padding=self._padding)

    def mask_sample(
        self,
        uv_co: Float[Tensor, "N 2"],
        mode: Optional[MaskMode] = None,
    ) -> Float[Tensor, "B N 1"]:
        """Bilinearly sample the mask at normalized coordinates.

        UV ``[0, 1]`` maps to the logical (unpadded) frame; values outside
        that range sample the implicit padding region.

        Args:
            uv_co: Normalized coordinates (N, 2) in pixel-center convention.
            mode: Reduction mode passed to :meth:`mask`. If None, uses the
                instance's ``mask_mode``.

        Returns:
            Sampled mask values (B, N, 1).
        """
        mask = self.mask(mode)
        sampled = ImgUtils.uv_sample(mask, uv_co, padding=self._padding)
        return sampled

    @torch.no_grad()
    def sample_px_coords(
        self,
        N: int,
        mode: Optional[MaskMode] = None,
        noise: bool = False,
    ) -> Float[Tensor, "N 2"]:
        """Sample N pixel coordinates weighted by the mask.

        Args:
            N: Number of coordinates to sample.
            mode: Reduction mode passed to :meth:`mask`. If None, uses the
                instance's ``mask_mode``.
            noise: If True, jitter each sampled coordinate uniformly within
                half a pixel on either side of its pixel center.

        Returns:
            Sampled coordinates (N, 2) in pixel-center convention.

        Notes:
            - Sampling uses ``torch.multinomial`` with replacement, so
              coordinates may repeat.
            - Coordinates follow the pixel-center convention of
              :meth:`ImgUtils.gen_px_coords` (centered at ``(i + 0.5) / D``).
            - With ``noise=True``, jitter is uniform in ``[-0.5/H, 0.5/H]``
              for y and ``[-0.5/W, 0.5/W]`` for x.
        """
        return ImgUtils.sample_px_coords(self.mask(mode), N, noise=noise)

    @torch.no_grad()
    def sample_points(
        self,
        num_points: int,
        mode: Optional[MaskMode] = None,
        noise: bool = False,
    ) -> Float[Tensor, "B num_points 2"]:
        """Sample point coordinates per batch element from the mask.

        Each pixel is treated as a Bernoulli trial whose success
        probability is the mask value. ``num_points`` coordinates are
        sampled per batch element (with replacement) and returned in the
        pixel-center convention used by :meth:`ImgUtils.gen_px_coords`.

        Args:
            num_points: Number of coordinates to sample per batch element.
            mode: Reduction mode passed to :meth:`mask`. If None, uses the
                instance's ``mask_mode``.
            noise: If True, jitter each sampled coordinate uniformly
                within half a pixel on either side of its center.

        Returns:
            Sampled coordinates (B, ``num_points``, 2) in pixel-center
            convention.

        Raises:
            ValueError: If the resolved mask contains negative values.
        """
        mask = self.mask(mode)  # (B, 1, H, W)
        B, _, H, W = mask.shape
        if (mask < 0).any():
            raise ValueError("sample_points expects non-negative mask values.")
        weights = mask.reshape(B, -1)  # (B, H*W)
        indices = torch.multinomial(
            weights, num_points, replacement=True
        )  # (B, num_points)
        co = ImgUtils.gen_px_coords(H, W, mask.device)  # (2, H, W)
        co_flat = co.reshape(2, -1).T  # (H*W, 2)
        sampled = co_flat[indices]  # (B, num_points, 2)
        if noise:
            sampled = sampled.clone()
            sampled[..., 0].add_(
                (torch.rand(B, num_points, device=mask.device) - 0.5) / H
            )
            sampled[..., 1].add_(
                (torch.rand(B, num_points, device=mask.device) - 0.5) / W
            )
        return sampled

    def extract_image_patches(
        self,
        patch_size: Optional[int],
        padding_mode: str = "replicate",
    ) -> Float[Tensor, "B P S C"]:
        """Extract non-overlapping patches.

        Args:
            patch_size: Patch size. None falls back to a single patch.
            padding_mode: Padding mode for F.pad.

        Returns:
            Image patches (B, P, S, C).
        """
        return ImgUtils.extract_image_patches(
            self._tensor,
            patch_size,
            padding_mode=padding_mode,
        )

    def pad(
        self,
        padding: Tuple[int, int, int, int],
        mode: Literal["constant", "reflect", "replicate", "circular"] = "replicate",
    ) -> Splimage:
        """Pad the image and update padding metadata. Returns new Splimage.

        Args:
            padding: Padding to add (top, bottom, left, right).
            mode: Padding mode passed to ``F.pad``. One of ``"constant"``,
                ``"reflect"``, ``"replicate"``, ``"circular"``.

        Returns:
            New Splimage with the padded tensor and updated padding.
        """
        new = self._pad_impl(padding, mode)
        return new

    def pad_(
        self,
        padding: Tuple[int, int, int, int],
        mode: Literal["constant", "reflect", "replicate", "circular"] = "replicate",
    ) -> Splimage:
        """Pad the image in-place and update padding metadata.

        Args:
            padding: Padding to add (top, bottom, left, right).
            mode: Padding mode passed to ``F.pad``. One of ``"constant"``,
                ``"reflect"``, ``"replicate"``, ``"circular"``.
        """
        pt, pb, pl, pr = padding
        self._tensor = F.pad(self._tensor, (pl, pr, pt, pb), mode=mode)
        self._padding = (
            self._padding[0] + pt,
            self._padding[1] + pb,
            self._padding[2] + pl,
            self._padding[3] + pr,
        )
        self._invalidate()
        return self

    def _pad_impl(
        self,
        padding: Tuple[int, int, int, int],
        mode: Literal["constant", "reflect", "replicate", "circular"] = "replicate",
    ) -> Splimage:
        pt, pb, pl, pr = padding
        padded = F.pad(self._tensor, (pl, pr, pt, pb), mode=mode)
        new = Splimage(padded)
        new._padding = (
            self._padding[0] + pt,
            self._padding[1] + pb,
            self._padding[2] + pl,
            self._padding[3] + pr,
        )
        return new

    def erode(self, padding: Tuple[int, int, int, int]) -> Splimage:
        """Crop the image by (top, bottom, left, right). Returns new Splimage.

        The inverse of :meth:`pad`: each entry trims that many pixels off
        the corresponding edge and the metadata is reduced accordingly.

        Args:
            padding: Amount to erode (top, bottom, left, right). Each
                entry must be ``>= 0`` and not exceed the current
                dimension minus the other axis entries.

        Raises:
            ValueError: If any entry is negative or exceeds the available
                pixels.
        """
        pt, pb, pl, pr = padding
        H, W = self.H, self.W
        if pt < 0 or pb < 0 or pl < 0 or pr < 0:
            raise ValueError(f"erode entries must be non-negative, got {padding}.")
        if pt + pb >= H:
            raise ValueError(
                f"Vertical erode {pt + pb} would consume the entire height {H}."
            )
        if pl + pr >= W:
            raise ValueError(
                f"Horizontal erode {pl + pr} would consume the entire width {W}."
            )
        cropped = self._tensor[:, :, pt : H - pb, pl : W - pr]
        new = Splimage(cropped)
        new._padding = (
            max(self._padding[0] - pt, 0),
            max(self._padding[1] - pb, 0),
            max(self._padding[2] - pl, 0),
            max(self._padding[3] - pr, 0),
        )
        return new

    def erode_(self, padding: Tuple[int, int, int, int]) -> Splimage:
        """Erode in-place.

        Args:
            padding: Amount to erode (top, bottom, left, right). Each
                entry must be ``>= 0`` and not exceed the current
                dimension minus the other axis entries.

        Raises:
            ValueError: If any entry is negative or exceeds the available
                pixels.
        """
        pt, pb, pl, pr = padding
        H, W = self.H, self.W
        if pt < 0 or pb < 0 or pl < 0 or pr < 0:
            raise ValueError(f"erode entries must be non-negative, got {padding}.")
        if pt + pb >= H:
            raise ValueError(
                f"Vertical erode {pt + pb} would consume the entire height {H}."
            )
        if pl + pr >= W:
            raise ValueError(
                f"Horizontal erode {pl + pr} would consume the entire width {W}."
            )
        self._tensor = self._tensor[:, :, pt : H - pb, pl : W - pr]
        self._padding = (
            max(self._padding[0] - pt, 0),
            max(self._padding[1] - pb, 0),
            max(self._padding[2] - pl, 0),
            max(self._padding[3] - pr, 0),
        )
        self._invalidate()
        return self
