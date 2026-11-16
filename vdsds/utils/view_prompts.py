"""View-dependent prompt embeddings: canonical-view caching and mixing.

Every concept prompt is expanded into one variant per canonical view
('front', 'side', 'back', 'top', 'bottom'), all variants are encoded once
up front, and each render pulls the composite embedding for its camera:
the concept's anchor-basis weight is split across its view sub-prompts by
the camera's view weights.

Exposes:
    CANONICAL_VIEWS: canonical camera-position directions.
    ViewPromptSet: the embedding cache + per-render composite builders.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from .spherical_basis import SphericalGaussianBasis

logger = logging.getLogger(__name__)

# Camera-position unit directions (from the object center toward the
# camera), in the renderer's world frame, validated against the cow model
# (~/3D/models/cow.glb): the '-Z' camera shows the face head-on, '+Z' the
# rear, the '-Y' camera the dorsal outline (the renderer's world up,
# UP = [0, -1, 0]), '+Y' the belly, and '+/-X' the profile flanks. 'side'
# is a single direction: both left and right cameras fall off into the
# same 'side view' prompt.
CANONICAL_VIEWS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 0.0, -1.0),
    "back": (0.0, 0.0, 1.0),
    "side": (1.0, 0.0, 0.0),
    "top": (0.0, -1.0, 0.0),
    "bottom": (0.0, 1.0, 0.0),
}


def canonical_directions(views: list[str]) -> Float[Tensor, "V 3"]:
    """Stacks the canonical camera-position directions for the named views.

    Args:
        views: Canonical view names (subset of CANONICAL_VIEWS, any order).

    Returns:
        out: Unit camera-position directions, one row per requested view.
    """
    unknown = [v for v in views if v not in CANONICAL_VIEWS]
    if unknown:
        raise ValueError(
            f"Unknown canonical view(s) {unknown}; choose from "
            f"{sorted(CANONICAL_VIEWS)}."
        )
    return torch.tensor([CANONICAL_VIEWS[v] for v in views], dtype=torch.float32)


def format_view_weights(
    weights: Float[Tensor, "V"],
    views: list[str],
    mode: str,
) -> str:
    """Formats canonical view weights as a compact overlay string.

    Args:
        weights: View weights for one direction (one row of
            ViewPromptSet.view_weights output).

    Returns:
        out: e.g. "view prompts (mix): front 0.79  side 0.11  top 0.05
        back 0.03  bottom 0.02".
    """
    parts = "  ".join(
        f"{name} {weight:.2f}" for name, weight in zip(views, weights.tolist())
    )
    return f"view prompts ({mode}): {parts}"


class ViewPromptSet:
    """Precomputed per-(concept, view) prompt embeddings and the mixing.

    Holds one text embedding for every (concept prompt, canonical view)
    pair, plus the static branch embeddings (base prompt and the BSD blank
    branch) and the uncond embedding, all encoded once at startup. At
    render time the camera's position direction produces view weights over
    the canonical directions (partition of unity), which combine with the
    per-concept branch weights into composite embeddings.

    Attributes:
        embeds: Cached (concept, view) embeddings, concept-major (stored
            in float32 regardless of the encoder dtype; the guidance UNet
            casts text embeds itself).
        static_embeds: Static branch embeddings (base prompt, blank ""),
            in the same branch order they appear after the concepts.
        uncond: The uncond/negative embedding.
        texts: The variant texts behind `embeds`, concept-major.
        static_texts: The texts behind `static_embeds`.
        views: Canonical view names (the V axis order of `embeds`).
        mode: 'mix' (smooth weighted blend) or 'select' (hard nearest
            canonical view).

    Construction:
        build(...) -> ViewPromptSet: encodes the variant and static texts
        through a DeepFloydGuidance-style `encode_text_2`.
    """

    def __init__(
        self,
        embeds: Float[Tensor, "P V T D"],
        static_embeds: Float[Tensor, "S T D"],
        uncond: Float[Tensor, "T D"],
        texts: list[str],
        static_texts: list[str],
        views: list[str],
        mode: str = "mix",
        sigma_overlap: float = 0.75,
    ):
        self.embeds = embeds.float()
        self.static_embeds = static_embeds.float()
        self.uncond = uncond.float()
        self.texts = texts
        self.static_texts = static_texts
        self.views = views
        self.mode = mode
        self.sigma_overlap = sigma_overlap
        # Fixed spherical Gaussians centered on the canonical directions:
        # reuses the gradient-safe dot-space falloff of
        # SphericalGaussianBasis; the query direction's normalized basis
        # values are exactly the desired view mix (equal front/side weight
        # at 45 degrees by symmetry, top/bottom entering via elevation).
        self._basis = SphericalGaussianBasis(
            num_funcs=len(views),
            num_dims=1,
            batch_size=1,
            centroids=canonical_directions(views),
            sigma_overlap=sigma_overlap,
            normalize=True,
        ).to(embeds.device)

    @classmethod
    def build(
        cls,
        df,
        variant_texts: list[str],
        static_texts: list[str],
        views: list[str],
        negative_prompt: str | list[str] | None = None,
        mode: str = "mix",
        sigma_overlap: float = 0.75,
    ) -> ViewPromptSet:
        """Encodes all prompt variants and static branches in one pass.

        Args:
            df: Diffusion guidance exposing `encode_text_2` (DeepFloydGuidance).
            variant_texts: Fully decorated concept-variant texts,
                concept-major: [c0v0, c0v1, ..., c1v0, ...].
            static_texts: Branch texts that do not vary with the view
                (base prompt, BSD blank branch), in branch order.
            views: Canonical view names, matching the variant grouping.
            negative_prompt: Uncond text (str or list); None encodes the
                empty prompt as uncond.
            mode: View-weight mode, 'mix' or 'select'.
            sigma_overlap: Angular falloff of the canonical view Gaussians.

        Returns:
            out: The populated cache (embeddings already on the encoder
            device).
        """
        texts = variant_texts + static_texts
        if negative_prompt is None:
            negatives = None
        elif isinstance(negative_prompt, str):
            negatives = [negative_prompt] * len(texts)
        else:
            negatives = list(negative_prompt)
        # batch_size=1: each text encoded once; encode_text_2 appends a
        # single neg row (pos.shape[0] // len(prompt) == 1).
        out = df.encode_text_2(texts, negative_prompt=negatives, batch_size=1)
        pos = out[: len(variant_texts)]  # (P*V, T, D)
        static_embeds = out[len(variant_texts) : len(texts)]  # (S, T, D)
        uncond = out[-1]  # (T, D)
        num_concepts, remainder = divmod(len(variant_texts), len(views))
        if remainder or num_concepts == 0:
            raise ValueError(
                f"variant_texts holds {len(variant_texts)} entries, not "
                f"divisible into whole concepts for {len(views)} view(s); "
                "variant texts must be concept-major with exactly one "
                "variant per (concept, view)."
            )
        embeds = pos.reshape(num_concepts, len(views), *pos.shape[1:])  # (P, V, T, D)
        return cls(
            embeds=embeds,
            static_embeds=static_embeds,
            uncond=uncond,
            texts=variant_texts,
            static_texts=static_texts,
            views=list(views),
            mode=mode,
            sigma_overlap=sigma_overlap,
        )

    @property
    def num_concepts(self) -> int:
        return self.embeds.shape[0]

    @property
    def num_static(self) -> int:
        return self.static_embeds.shape[0]

    def view_weights(self, dirs: Float[Tensor, "B 3"]) -> Float[Tensor, "B V"]:
        """Weights over the canonical views for camera position directions.

        Args:
            dirs: Unit directions from the object center toward each
                camera (same frame as CANONICAL_VIEWS); a single (3,)
                direction is treated as a one-element batch.

        Returns:
            out: Per-render view weights; 'mix' rows sum to 1, 'select'
            rows are one-hot at the nearest canonical view.
        """
        if dirs.ndim == 1:
            dirs = dirs[None]
        dirs = dirs.reshape(-1, 3)  # drop stray leading dims
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        raw = self._basis._basis(dirs)  # (B, V)
        if self.mode == "mix":
            return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.mode == "select":
            onehot = torch.zeros_like(raw)
            onehot.scatter_(-1, raw.argmax(dim=-1, keepdim=True), 1.0)
            return onehot
        raise ValueError(f"View mode '{self.mode}' is invalid (use mix/select).")

    def composite(
        self,
        dirs: Float[Tensor, "B 3"],
        concept_weights: Float[Tensor, "B P"] | None = None,
    ) -> Float[Tensor, "B T D"]:
        """Composite embeddings for a render batch.

        Each concept's weight (from the anchor basis; uniform ones when
        absent) is split across its view sub-prompts by the camera's view
        weights, and the weighted cache rows are summed.

        Args:
            dirs: Unit camera-position directions (B, 3).
            concept_weights: Per-concept branch weights; None mixes all
                concepts uniformly.

        Returns:
            out: One composite positive embedding per render.
        """
        weights = self.view_weights(dirs)  # (B, V)
        if concept_weights is None:
            # Uniform average over the concepts keeps the composite a
            # convex combination (relevant for multi-prompt SDS; BSD and
            # single-prompt SDS are unaffected).
            concept_weights = torch.full(
                (weights.shape[0], self.num_concepts),
                1.0 / self.num_concepts,
                dtype=weights.dtype,
                device=weights.device,
            )
        # (B,P) x (B,V) x (P,V,T,D) -> (B,T,D)
        return torch.einsum(
            "bp,bv,pvtd->btd", concept_weights, weights, self.embeds
        )

    def sds_embeds(
        self,
        dirs: Float[Tensor, "B 3"],
        concept_weights: Float[Tensor, "B P"] | None = None,
    ) -> Float[Tensor, "2B T D"]:
        """SDS layout: per-render composites followed by uncond rows.

        Matches df.SDS's expectation of [pos rows; uncond rows] for its
        CFG chunking (the first half feeds the pos latents, the second
        half the duplicated uncond latents).

        Args:
            dirs: Unit camera-position directions (B, 3).
            concept_weights: Per-concept branch weights; None mixes all
                concepts uniformly.

        Returns:
            out: Text embeddings, first B rows composite, last B rows the
            uncond embedding.
        """
        mix = self.composite(dirs, concept_weights)  # (B, T, D)
        uncond = self.uncond[None].expand(mix.shape[0], -1, -1)  # (B, T, D)
        return torch.cat([mix, uncond], dim=0)

    def bsd_embeds(
        self,
        dirs: Float[Tensor, "B 3"],
        attn_alphas: Float[Tensor, "B P-1"],
        num_funcs: int,
    ) -> Float[Tensor, "PpBplusB T D"]:
        """BSD branch layout: per-view composite embeddings per branch.

        Each concept branch's embedding becomes that branch's
        view-composite (the branch's anchor alpha distributed over the
        concept's view sub-prompts); the static branches (base prompt,
        BSD blank branch) keep their fixed embeddings. Row order matches
        encode_text_2's text-major repetition: [c0 x B, c1 x B, ...,
        static x B, uncond x B], so the ActvnReplace branch/alpha pairing
        is unchanged.

        Args:
            dirs: Unit camera-position directions (B, 3).
            attn_alphas: The get_prompt_basis output (B, num_funcs +
                base?): concept columns first, then the optional base
                weight column; the blank branch carries no alpha.
            num_funcs: Number of concept branches (leading alpha columns).

        Returns:
            out: Text embeddings for ActvnReplace, uncond rows last.
        """
        expected = num_funcs + self.num_static - 1
        if attn_alphas.shape[1] != expected:
            raise ValueError(
                f"attn_alphas holds {attn_alphas.shape[1]} column(s) but "
                f"{num_funcs} concept branch(es) plus {self.num_static} "
                f"static branch(es) (minus the blank branch) expect "
                f"{expected}."
            )
        concept_weights = attn_alphas[:, :num_funcs]  # (B, F)
        branches = torch.einsum(
            "bf,bv,fvtd->fbtd",
            concept_weights,
            self.view_weights(dirs),  # (B, V)
            self.embeds,
        )  # (F, B, T, D)
        static = self.static_embeds[:, None].expand(
            self.num_static, dirs.shape[0], *self.static_embeds.shape[1:]
        )  # (S, B, T, D)
        full = torch.cat([branches, static], dim=0)  # (P', B, T, D)
        rows = full.reshape(-1, *full.shape[2:])  # (P'*B, T, D), branch-major
        uncond = self.uncond[None].expand(dirs.shape[0], -1, -1)  # (B, T, D)
        return torch.cat([rows, uncond], dim=0)

    def to_dict(self) -> dict[str, object]:
        """Serializes the cache into a plain dict of tensors and lists."""
        return {
            "embeds": self.embeds.cpu(),
            "static_embeds": self.static_embeds.cpu(),
            "uncond": self.uncond.cpu(),
            "texts": list(self.texts),
            "static_texts": list(self.static_texts),
            "views": list(self.views),
            "mode": self.mode,
            "sigma_overlap": self.sigma_overlap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ViewPromptSet:
        """Rebuilds a cache from `to_dict` output."""
        return cls(
            embeds=data["embeds"],
            static_embeds=data["static_embeds"],
            uncond=data["uncond"],
            texts=list(data["texts"]),
            static_texts=list(data["static_texts"]),
            views=list(data["views"]),
            mode=data.get("mode", "mix"),
            sigma_overlap=float(data.get("sigma_overlap", 0.75)),
        )

    def save(self, path: str | Path) -> None:
        """Persists the cache to a .pt file for inspection/reuse."""
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.to_dict(), path)
        logger.info(f"Saved view prompt cache ({self.texts[0]}, ...) to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ViewPromptSet:
        """Loads a cache saved by :meth:`save`."""
        return cls.from_dict(torch.load(path, weights_only=False))
