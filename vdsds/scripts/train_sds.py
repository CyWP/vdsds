import logging
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import torch
from jaxtyping import Float
from torch import Tensor

from ..rendering import Camera, CameraCoordinates, LightSource, Renderer
from ..rendering.light import cosine_sample_hemisphere
from ..rendering.shader import (
    Albedo,
    Alpha,
    Antialias,
    BackgroundColor,
    Clamp,
    HardLambertShader,
    LambdaShader,
    Normal,
    SoftLambertShader,
)
from ..utils.config import Config
from ..utils.deepfloyd import DeepFloydGuidance
from ..constraints import (
    Constraint,
    ConstraintManager,
    anchors_from_constraints,
    prompts_from_constraints,
)
from ..utils.depth import DepthAnything
from ..utils.img import ImgUtils, Splimage
from ..utils.quaternion import Quaternion
from ..utils.resize_right import resize
from ..utils.spherical_basis import SphericalGaussianBasis
from .analyze import AnalyzeDeformation
from .base import ViewableScript
from .orbit import OrbitFrames

logger = logging.getLogger(__name__)


class TrainModelSDS(ViewableScript):
    # Shading profiles place the shader/lighting keys that differ between the
    # legacy renderer setup and the MeshUp-parity renderer setup (see
    # experiments/meshup_parity.yaml). Profile values only fill leaves that
    # neither the class defaults nor the user config set, so explicitly
    # configured values always win. Select 'none' to skip the mechanism
    # entirely (all profile-managed keys must then come from the config).
    _shading_profiles: ClassVar[dict[str, dict[str, any]]] = {
        "legacy": {
            "shader": {
                "shading": "soft",
                "bg_color": [0.2, 0.7, 0.0],
                "crop_fit": True,
            },
            "lighting": {
                "alignment": "camera",
                "jitter_sigma": math.pi / 4,
            },
        },
        "meshup": {
            "shader": {
                "shading": "hard",
                "bg_color": [1.0, 1.0, 1.0],
                "crop_fit": False,
            },
            "lighting": {
                "alignment": "cosine_camera",
                "jitter_sigma": 0.0,
            },
        },
    }
    _default_profile: ClassVar[str] = "meshup"  # New (workspace) config.

    _default_config_overrides: ClassVar[dict[str, any]] = {
        "log": {
            "print": True,
            "save": True,
        },
        "window": {
            "fps": 12,
            "close_on_finish": True,
        },
        "path": {
            "run_dir": None,
        },
        "optim": {
            "epochs": 1500,
            "lr": 0.0025,
            "accum_steps": 2,
            "step_mode": "accum",  # Options: 'accum' (ours), 'meshup'
            "seed": None,
            "train_model": False,
            "train_centroids": False,
            "train_sigmas": False,
            "train_weights": True,
        },
        "diffusion": {
            "loss": "bsd",  # Options: 'sds','bsd'
            "model_size": "L",  # Options: 'S', 'M', 'L', 'XL'
            "dtype": "float16",
            "cpu_offload": False,
            "guidance_scale": 100.0,
        },
        "prompt": {
            "text": ["Moose with large antlers", "Giraffe"],
            "anchors": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            # "text": ["Moose with large antlers"],
            # "anchors": [[1.0, 0.0, 0.0]],
            "overlap": 0.75,
            "basis_proc": "norm",  # Options: 'norm', 'square', 'max'
            "negative": None,
            "base": None,  # Base prompt text, e.g. the base object.
            "base_branch_weight": -0.2,  # Blend weight of the base branch.
            # Extra strings appended to every prompt (e.g. rendering style
            # qualifiers like '4k', 'realistic', 'unreal engine').
            "extra": [],
        },
        "loss": {
            "diffusion": 1.0,
            "jacobian": 0.0,  # Weight-space prior; swept to suppress motion.
            "composed_jacobian": 500.0,  # Per-entry reg at rendered view dirs.
            "laplacian": 0.0,
            "volume": 0.0,
            "volume_eval_points": 32,
            "view_dependence": 3000.0,  # Anti-inflation of the view basis.
            # Constraint-view rendered-vs-monocular depth distribution
            # loss over the constraint images' DA v3 depths (see
            # TrainModelSDS.depth_loss; wired as an epoch regularizer;
            # DA v3 load/infer/unload happens once before the run).
            "depth": 0.0,
            # Constraint-view silhouette consistency loss (top-k
            # Chamfer against the constraint overlays' alpha; wired as
            # an epoch regularizer).
            "silhouette": 0.0,
            # Weight of the silhouette under-coverage (inverse) term;
            # 0 keeps the original over-coverage-only behavior.
            "silhouette_inverse": 0.0,
        },
        "model": {"name": "mesh"},
        "deformation": {
            "name": "vd_full",  # Options: 'full', 'vd_full', 'vd_restrained'
            "num_funcs": 12,
            "init": "fibonacci",  # Options: 'fibonacci', 'random'
            "overlap": 2.25,
            "normalize": False,
            "equi_optim_steps": 200,
        },
        "camera": {
            "H": 512,
            "W": 512,
            "Zn": 0.01,
            "Zf": 50.0,
            "F_min": 30,
            "F_max": 90,
            "radius_min": 1.25,
            "radius_max": 1.75,
            "views": 8,
            "view_source": "orbit",  # Options: 'random', 'orbit'abs
            "elevation_min": 0.0,  # Only used for 'orbit' mode
            "elevation_max": math.pi / 3,  # Only used for 'orbit' mode
            "point_upwards": True,
            "aug_views": 0,
            "aug_type": "jitter",  # Options: 'jitter', 'random'
            "jitter_range": math.pi / 4,
            "adapt_dist": False,  # MeshUp-style per-epoch distance rescale.
            "force_deformation_alignment": False,
        },
        "shader": {
            "resize": True,
            "down_H": 224,  # Only used if 'resize' is True
            "down_W": 224,  # Only used if 'resize' is True
            "shading_beta": 5.0,
            # 'shading', 'bg_color' and 'crop_fit' are managed by the
            # shading profiles (see _shading_profiles).
            "antialias": True,
        },
        "lighting": {
            # 'alignment' and 'jitter_sigma' are managed by the shading
            # profiles (see _shading_profiles).
            "strength": 5.0,
        },
        "post": {
            "orbit": True,
            "check_batch_views": 2,
            "analyze": True,
        },
        # Training-phase schedule: switches the shading/lighting profile,
        # the step mode ('accum' vs 'meshup' batch computing) and any other
        # dotted-key overrides at epoch boundaries. Each entry is at least
        # {'epochs': N}; 'profile': 'meshup'|'legacy'|'none' applies its
        # shading profile, remaining keys are merged as full-strength
        # overrides (e.g. {'optim.step_mode': 'meshup'}). When non-empty,
        # the phase epoch counts sum to the effective total and replace
        # 'optim.epochs'.
        "schedule": [],
    }

    def _apply_profile(self) -> None:
        """Applies the shading profile selected by the merged config.

        Fills every profile-managed leaf key that neither the class
        defaults nor the user config set, so explicitly configured values
        always win. Runs once at the top of :meth:`run`, before any
        rendering settings are read; the whole mechanism can be bypassed
        with 'profile: none'.
        """
        name = self.config.pop("profile", None)
        if name is None:
            name = self._default_profile
        if name == "none":
            return
        if name not in self._shading_profiles:
            raise ValueError(f"Shading profile '{name}' is invalid.")
        self._fill_missing(self.config, self._shading_profiles[name])

    def _fill_missing(self, config: Config, values: dict[str, any]):
        for key, value in values.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, Mapping):
                self._fill_missing(config[key], value)

    def get_constraints(self) -> dict[str, Constraint]:
        """Load the constraints from the configured JSON file.

        Reads the JSON at 'path.constraints' (typically provided via the
        --constraints flag). Returns empty when the config entry is None
        (the default); constraints are only ever loaded from the file.

        Returns:
            out: The id-keyed constraint dict.
        """
        path = self.config.get("path", None)
        if path is None:
            return {}
        path = path.get("constraints", None)
        if path is None:
            return {}
        return ConstraintManager(Path(path)).items()

    def get_depth_renderer(self) -> Renderer:
        """Cached depth renderer (clip-space z/w, ``NO_DEPTH`` background)."""
        if not hasattr(self, "_depth_renderer"):
            self._depth_renderer = Renderer.depth(self.device)
        return self._depth_renderer

    def prepare_constraint_depths(self) -> None:
        """Precompute the constraint mono depths before the run starts.

        DA v3 loads, infers all constraint images and unloads within
        the single :meth:`DepthAnything.depth_from_constraints` call;
        nothing of it is needed during training.
        """
        self._mono_depths = DepthAnything.depth_from_constraints(self.constraints)

    def depth_loss(self, mode: str = "affine") -> Float[Tensor, ""]:
        """Rendered-vs-monocular depth distribution loss over the views
        of the loaded constraints.

        The monocular references are the precomputed DA v3 depths of
        the constraint images (:meth:`prepare_constraint_depths`, run
        once before training when ``loss.depth`` is enabled). The
        rendered reference is the clip-space depth of the deformed
        model at the constraint cameras (``Renderer.depth``,
        ``NO_DEPTH`` background); mono maps are resampled to the
        camera resolution.

        Args:
            mode: Depth loss mode ('affine' or 'minmax') passed
                through to :func:`depth_loss`.

        Returns:
            out: Scalar depth loss over the constraint batch.
        """
        from ..utils.depth import depth_loss

        if not hasattr(self, "_mono_depths"):
            raise ValueError(
                "Constraint mono depths are not precomputed; enable "
                "'loss.depth' so they are derived before the run starts."
            )
        rendered, measured = [], []
        for cid, mono in self._mono_depths.items():
            camera = self.constraints[cid].view.camera.copy().to(self.device)
            data = self.model.deformed(camera)
            rendered.append(
                self.get_depth_renderer()(
                    data, camera, self.randomized_light(camera)
                )["depth"]
            )
            measured.append(
                ImgUtils.resize(mono[None, None], camera.H, camera.W).to(
                    self.device
                )
            )
        if len(rendered) == 0:
            raise ValueError(
                "Depth loss requires constraints with images and views."
            )
        return depth_loss(
            torch.cat(rendered, dim=0), torch.cat(measured, dim=0), mode=mode
        )

    def silhouette_loss(self) -> Float[Tensor, ""]:
        """Silhouette consistency loss over the views of the loaded
        constraints.

        Forwards to the original ported implementation
        (:func:`vdsds.utils.silhouette.silhouette_loss`, unmodified:
        top-k Chamfer distance from projected out-of-mask deformed-mesh
        vertices to the target silhouette pixels, over-coverage term
        only). The loss requires an RGBA image from the constraint: the
        constraint overlay is placed at the constraint camera's
        resolution and its alpha channel carries the target silhouette.

        Constraints without an overlay are skipped with a notice; with
        none qualifying, this raises. Wired into the epoch update as a
        regularizer via ``_regularizer_losses``; the config multiplier
        ``loss.silhouette`` applies there (per epoch; no
        accumulation-step compensation since the constraint batch is
        constant).

        Returns:
            out: Scalar silhouette loss over the constraint batch.
        """
        from ..utils.silhouette import silhouette_loss as silhouette_loss_fn

        tgt_imgs, meshes, cameras = [], [], []
        for cid, constraint in self.constraints.items():
            view = constraint.view if constraint is not None else None
            overlay = view.overlay if view is not None else None
            camera = view.camera if view is not None else None
            if overlay is None:
                print(f"Skipping constraint {cid[:6]}: no overlay (rgba target).")
                continue
            if camera is None:
                print(f"Skipping constraint {cid[:6]}: no view attached.")
                continue
            camera = camera.copy().to(self.device)
            tgt_imgs.append(overlay.image(camera.H, camera.W).to(self.device))
            meshes.append(self.model.deformed(camera))
            cameras.append(camera)
        if len(tgt_imgs) == 0:
            raise ValueError("Silhouette loss requires constraints with overlays.")
        loss_cfg = self.config.get("loss", None)
        in_weight = (
            loss_cfg.get("silhouette_inverse", 0.0) if loss_cfg is not None else 0.0
        )
        return silhouette_loss_fn(
            torch.cat(tgt_imgs, dim=0),
            meshes,
            cameras,
            in_weight=in_weight,
        )

    def run(self):
        device = self.device
        config = self.config
        self._apply_profile()
        self.constraints = self.get_constraints()
        if self.constraints:
            print(f"Loaded {len(self.constraints)} constraint(s) for training.")
        if self.constraints and config.loss.depth != 0.0:
            # DA v3 load/infer/unload happens inside this single call,
            # before training starts; nothing of it runs per epoch.
            self.prepare_constraint_depths()
        df = DeepFloydGuidance(config.diffusion, device)
        seed = config.optim.seed
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        self._radius_min = config.camera.radius_min
        self._radius_max = config.camera.radius_max
        self.model.train(
            model=config.optim.train_model,
            centroids=config.optim.train_centroids,
            sigmas=config.optim.train_sigmas,
            weights=config.optim.train_weights,
        )
        self.renderer = self.get_renderer()
        accum_steps = config.optim.accum_steps
        prompts, text_embeds, prompt_basis = self.get_text_embeds(df)
        batch_size = config.camera.views * (1 + config.camera.aug_views)
        optimizer = torch.optim.Adam(
            [*self.model.get_parameters()],
            lr=config.optim.lr,
        )
        phases = self._build_schedule()
        epoch = 0
        for phase_epochs, phase in phases:
            if phase is not None:
                self._enter_phase(optimizer, phase, epoch)
            for _ in range(phase_epochs):
                if self.abort_requested():
                    return
                optimizer.zero_grad()
                if config.camera.adapt_dist:
                    self._adapt_dist(epoch)
                cameras = self.camera_batch()
                loss_dict = self.epoch_update(
                    optimizer, cameras, df, text_embeds, prompt_basis, len(prompts)
                )
                self.log(epoch, loss_dict)
                epoch += 1

    def _build_schedule(self) -> list[tuple[int, dict[str, any] | None]]:
        """Resolves the phase schedule into (epochs, overrides) pairs.

        An empty 'schedule' config yields the single implicit phase of
        'optim.epochs' epochs without overrides, matching the unscheduled
        behavior. Otherwise each phase contributes its 'epochs' count (all
        counts must be positive, summing to the effective total) together
        with its override dict (minus 'epochs'), and 'optim.epochs' is set
        to the total so saved configs record the true run length.
        """
        schedule = self.config.get("schedule") or []
        if not schedule:
            return [(int(self.config.optim.epochs), None)]
        phases: list[tuple[int, dict[str, any]]] = []
        for phase in schedule:
            phase = dict(phase)  # do not mutate the config's dict entries
            epochs = int(phase.pop("epochs", 0))
            if epochs <= 0:
                raise ValueError(
                    f"Schedule phase expects a positive 'epochs': {phase}"
                )
            phases.append((epochs, phase))
        self.config.optim.epochs = sum(n for n, _ in phases)
        return phases

    def _enter_phase(
        self, optimizer: torch.optim.Optimizer, phase: dict[str, any], epoch: int
    ) -> None:
        """Switches to a training phase at an epoch boundary.

        Applies the named shading profile first ('profile' is popped; its
        managed shader/lighting values are set, overriding the currently
        merged ones), then merges the remaining dotted-key overrides at full
        strength. 'optim.lr' propagates to the optimizer parameter groups
        (Adam is constructed once for the whole run). The renderer is
        rebuilt so the shader chain picks up the new shading mode.
        """
        profile = phase.pop("profile", None)
        if profile is not None and profile != "none":
            if profile not in self._shading_profiles:
                raise ValueError(f"Shading profile '{profile}' is invalid.")
            for section, values in self._shading_profiles[profile].items():
                self.config[section].update(values)
        self.config.update(phase)
        # LR may arrive nested (Dict wrapping splits dotted keys) or dotted.
        lr = phase.get("optim.lr")
        if lr is None and isinstance(phase.get("optim"), Mapping):
            lr = phase["optim"].get("lr")
        if lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr
        if profile is not None or phase:
            self.renderer = self.get_renderer()
        logged: dict[str, any] = {"phase_profile": profile}
        logged.update({f"phase:{key}": value for key, value in phase.items()})
        self.log(epoch, logged)

    def epoch_update(
        self,
        optimizer: torch.optim.Optimizer,
        cameras: list[tuple[Camera, Camera]],
        df: DeepFloydGuidance,
        text_embeds,
        prompt_basis: SphericalGaussianBasis | None,
        num_prompts: int,
    ) -> dict[str, any]:
        """Performs one epoch's parameter update with the configured mode.

        Dispatches on `optim.step_mode`. The optimizer gradients must be
        zeroed before the call (one `zero_grad` per epoch for both modes).

        Args:
            optimizer: Adam instance over the model parameters.
            cameras: Camera pairs (render, deformation) of this epoch.
            df: Diffusion guidance.
            text_embeds: Pre-encoded prompt text embeddings.
            prompt_basis: The BSD prompt basis (None for 'sds' loss).
            num_prompts: Number of concept prompts.

        Returns:
            out: Loss dict for logging.
        """
        mode = self.config.optim.step_mode
        if mode == "accum":
            return self._epoch_accum(
                optimizer, cameras, df, text_embeds, prompt_basis, num_prompts
            )
        elif mode == "meshup":
            return self._epoch_meshup(
                optimizer, cameras, df, text_embeds, prompt_basis, num_prompts
            )
        raise ValueError(f"Step mode '{mode}' is invalid.")

    def _epoch_accum(
        self,
        optimizer: torch.optim.Optimizer,
        cameras: list[tuple[Camera, Camera]],
        df: DeepFloydGuidance,
        text_embeds,
        prompt_basis: SphericalGaussianBasis | None,
        num_prompts: int,
    ) -> dict[str, any]:
        """Our epoch update: mean over accum_steps, one optimizer step.

        Renders and re-noises fresh at every accumulation step; all
        gradients are evaluated at the current parameters and the diffusion
        loss is divided by accum_steps so the accumulated diffusion
        gradient is the mean over the accumulation draws. Regularizers are
        applied once, against the batch-mean diffusion gradient, and a
        single `optimizer.step()` completes the epoch.

        Args:
            optimizer: Adam instance over the model parameters.
            cameras: Camera pairs (render, deformation) of this epoch.
            df: Diffusion guidance.
            text_embeds: Pre-encoded prompt text embeddings.
            prompt_basis: The BSD prompt basis (None for 'sds' loss).
            num_prompts: Number of concept prompts.

        Returns:
            out: Loss dict for logging.
        """
        accum_steps = self.config.optim.accum_steps
        loss_dict = {}
        epoch_loss = 0.0
        for _ in range(accum_steps):
            reconstruction_loss = (
                self.diffusion_loss(
                    cameras, df, text_embeds, prompt_basis, num_prompts
                )
                * self.config.loss.diffusion
            ) / accum_steps
            reconstruction_loss.backward()
            epoch_loss += reconstruction_loss.item()
        loss_dict["reconstruction_loss"] = epoch_loss
        self._apply_regularizers(cameras, loss_dict)
        optimizer.step()
        return loss_dict

    def _epoch_meshup(
        self,
        optimizer: torch.optim.Optimizer,
        cameras: list[tuple[Camera, Camera]],
        df: DeepFloydGuidance,
        text_embeds,
        prompt_basis: SphericalGaussianBasis | None,
        num_prompts: int,
    ) -> dict[str, any]:
        """MeshUp's epoch update: step inside the accumulation loop.

        Replicates the stepping pattern of MeshUp's `loop.py`: the render
        batch is computed once per epoch and reused (with `retain_graph`)
        across the accumulation iterations, the diffusion loss is applied
        at full weight (no 1/accum_steps normalization), regularizers are
        recomputed and backward-ed inside the loop, and `optimizer.step()`
        runs after every backward — two smaller steps per epoch whose
        second one contains the leftover gradients of the first (there is
        no zero_grad between the steps).

        Args:
            optimizer: Adam instance over the model parameters.
            cameras: Camera pairs (render, deformation) of this epoch.
            df: Diffusion guidance.
            text_embeds: Pre-encoded prompt text embeddings.
            prompt_basis: The BSD prompt basis (None for 'sds' loss).
            num_prompts: Number of concept prompts.

        Returns:
            out: Loss dict for logging.
        """
        accum_steps = self.config.optim.accum_steps
        loss_dict = {}
        epoch_loss = 0.0
        renders = self.get_training_renders(cameras)
        for _ in range(accum_steps):
            reconstruction_loss = self.diffusion_step(
                renders, cameras, df, text_embeds, prompt_basis, num_prompts
            ) * self.config.loss.diffusion
            # Retain: the next accumulation step backwards through the
            # same (render-time) graph — the stale-graph behavior of the
            # original loop.
            reconstruction_loss.backward(retain_graph=True)
            epoch_loss += reconstruction_loss.item()
            self._apply_regularizers(cameras, loss_dict)
            optimizer.step()
        loss_dict["reconstruction_loss"] = epoch_loss
        return loss_dict

    def _regularizer_losses(
        self, cameras: list[tuple[Camera, Camera]], accum_steps: int
    ) -> dict[str, tuple[torch.Tensor, float]]:
        """Builds the configured regularizer losses with their weights.

        Args:
            cameras: Camera pairs (render, deformation) of this epoch
                (needed for the composed-jacobian loss).
            accum_steps: Accumulation steps, used by the accum_steps-
                compensated regularizers.

        Returns:
            out: Map from log name (without '_loss' suffix) to
                (unweighted loss tensor, weight) pairs, in application
                order.
        """
        cfg = self.config
        regs: dict[str, tuple[torch.Tensor, float]] = {}
        if cfg.loss.jacobian != 0.0:
            # num_funcs compensates the basis-mean dilution of the
            # weight-space reg (per composed jacobian entry the realized
            # balance is loss.jacobian : diffusion). No accum_steps
            # scaling: the diffusion gradient is already a batch mean.
            regs["jacobian"] = (
                self.jacobian_loss(),
                cfg.loss.jacobian * cfg.deformation.num_funcs,
            )
        if cfg.loss.composed_jacobian != 0.0:
            # No num_funcs/accum_steps compensation: the mean is taken
            # over composed per-view jacobian entries and over the view
            # batch, matching the batch-mean diffusion gradient, so the
            # nominal weight is directly comparable to the per-entry
            # identity pull of a plain jacobian field.
            regs["composed_jacobian"] = (
                self.composed_jacobian_loss(cameras),
                cfg.loss.composed_jacobian,
            )
        if cfg.loss.view_dependence != 0.0:
            regs["view_dependence"] = (
                self.view_dependence_loss(),
                cfg.loss.view_dependence * accum_steps,
            )
        if cfg.loss.laplacian != 0.0:
            regs["laplacian"] = (
                self.laplacian_loss(),
                cfg.loss.laplacian * accum_steps,
            )
        if cfg.loss.volume != 0.0:
            regs["volume"] = (
                self.volume_loss(),
                cfg.loss.volume * accum_steps,
            )
        # Constraint-view steps: evaluated once per epoch on the fixed
        # constraint set, so no accumulation-step compensation.
        if cfg.loss.silhouette != 0.0:
            regs["silhouette"] = (self.silhouette_loss(), cfg.loss.silhouette)
        if cfg.loss.depth != 0.0:
            regs["depth"] = (self.depth_loss(), cfg.loss.depth)
        return regs

    def _apply_regularizers(
        self, cameras: list[tuple[Camera, Camera]], loss_dict: dict[str, any]
    ):
        """Backwards the configured regularizer losses and logs them.

        Args:
            cameras: Camera pairs (render, deformation) of this epoch.
            loss_dict: Log dict; regularizer entries are appended as
                '<name>_loss'.
        """
        for name, (loss, weight) in self._regularizer_losses(
            cameras, self.config.optim.accum_steps
        ).items():
            weighted = loss * weight
            weighted.backward()
            loss_dict[f"{name}_loss"] = weighted.item()

    def get_constraint_prompts(self) -> tuple[list[str], list[list[float]]]:
        """Prompts and their anchors from the loaded constraints.

        Constraints always carry views, so every prompt-bearing
        constraint has an anchor: the (prompt, anchor) lists are
        aligned, one entry per prompt-bearing constraint. Constraints
        without a prompt are simply not taken into account.

        Returns:
            out: (prompts, anchors) with one normalized (3,) anchor per
                prompt.
        """
        prompt_map = prompts_from_constraints(self.constraints)
        anchor_map = anchors_from_constraints(
            self.constraints, self.model.model.centroid
        )
        prompts = list(prompt_map.values())
        anchors = [
            [float(v) for v in anchor_map[cid].tolist()] for cid in prompt_map
        ]
        return prompts, anchors

    def get_text_embeds(
        self, df: DeepFloydGuidance
    ) -> tuple[list[str], Float[Tensor, "..."], SphericalGaussianBasis | None]:
        cfg = self.config
        txt = cfg.prompt.text
        anchors = cfg.prompt.anchors
        if self.constraints:
            print(
                "Deriving prompt(s)/anchor(s) from the loaded constraints; "
                "config prompt text/anchors are ignored."
            )
            txt, anchors = self.get_constraint_prompts()
        views = cfg.camera.views
        aug_views = cfg.camera.aug_views
        batch_size = views * (1 + aug_views)
        # prompt.extra qualifiers are appended verbatim to every prompt.
        # Accept a single string as one qualifier (CLI overrides arrive as
        # strings; lists survive the JSON parse path).
        extra = cfg.prompt.extra
        if isinstance(extra, str):
            extra = [extra.strip()] if extra.strip() else []
        extra_suffix = (
            ", " + ", ".join(str(e) for e in extra) if len(extra) > 0 else ""
        )
        if isinstance(txt, list):
            # MeshUp convention: the last prompt is kept raw (no rendering
            # suffix); extras still apply to every prompt.
            prompts = [
                p + ", a 3d rendering" + extra_suffix
                if i != len(txt) - 1
                else p + extra_suffix
                for i, p in enumerate(txt)
            ]
        else:
            prompts = [txt + ", a 3d rendering" + extra_suffix]
        if cfg.diffusion.loss == "bsd":
            # bsd blends one basis function per concept prompt; strings count
            # as a single prompt (len() would count characters).
            num_text = len(txt) if isinstance(txt, list) else 1
            centroids = torch.tensor(anchors, dtype=torch.float32)
            # A single anchor direction is commonly written flat ([x, y, z]);
            # normalize to the (N, 3) convention, then validate against the
            # function count with a useful message.
            if centroids.ndim == 1:
                centroids = centroids[None]
            if centroids.shape[0] != num_text:
                raise ValueError(
                    f"prompt.anchors holds {centroids.shape[0]} anchor(s) "
                    f"but the prompt text encodes {num_text} concept "
                    "prompt(s); provide one (x, y, z) anchor per prompt, "
                    "e.g. [[0.0, 1.0, 0.0]] for a single prompt."
                )
            basis = SphericalGaussianBasis(
                num_funcs=num_text,
                num_dims=1,
                batch_size=1,
                sigma_overlap=cfg.prompt.overlap,
                centroids=centroids,
                normalize=True,
            ).to(self.device)
            # Base prompt branch sits after the concept prompts and before
            # the empty uncond branch, so branch order matches the basis
            # columns plus the appended base weight.
            if cfg.prompt.base is not None:
                prompts.append(cfg.prompt.base + ", a 3d rendering" + extra_suffix)
            prompts.append("")
        else:
            basis = None
        if cfg.prompt.negative is None:
            # encode_prompt fills uncond tokens with [""] * batch internally.
            negative_prompt = None
        else:
            # encode_prompt requires the negative list to match the prompt
            # list length (each text is repeated batch_size times there);
            # encode_text_2 slices the embeds back down to batch_size.
            negative_prompt = [cfg.prompt.negative] * (
                len(prompts) * batch_size
            )  # (L * batch_size,)
        text_embeds = df.encode_text_2(
            prompts,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
        )
        return prompts, text_embeds, basis

    def get_prompt_basis(
        self, basis_func: SphericalGaussianBasis, deltas: Float[Tensor, "N 3"]
    ) -> Float[Tensor, "M N"]:
        cfg = self.config.prompt
        proc = cfg.basis_proc
        basis = basis_func._basis(deltas)
        if proc == "norm":
            basis = basis / basis.sum(dim=-1, keepdim=True)
        elif proc == "square":
            basis = basis**2
            basis = basis / basis.sum(dim=-1, keepdim=True)
        elif proc == "max":
            processed = torch.zeros_like(basis)
            # Advanced indexing over dim 0 would zero out entire rows:
            # scatter one column index per row explicitly.
            arange = torch.arange(basis.shape[0], device=basis.device)
            processed[arange, basis.argmax(dim=-1)] = 1.0
            basis = processed
        else:
            raise ValueError(f"Basis processign option '{proc}' is invalid.")

        if cfg.base is not None:
            # Append the constant base branch weight, then rescale the
            # blend weights so they sum to 1 per view.
            device = deltas.device
            base = torch.full(
                (basis.shape[0], 1),
                float(cfg.base_branch_weight),
                dtype=basis.dtype,
                device=device,
            )
            basis = torch.cat([basis, base], dim=-1)
            basis = basis / basis.sum(dim=-1, keepdim=True)

        return basis

    def diffusion_loss(
        self,
        cameras: list[tuple[Camera, Camera]],
        df: DeepFloydGuidance,
        text_embeds,
        alpha_basis: SphericalGaussianBasis | None,
        num_prompts: int,
    ) -> Float[Tensor, ""]:
        renders = self.get_training_renders(cameras)
        return self.diffusion_step(
            renders, cameras, df, text_embeds, alpha_basis, num_prompts
        )

    def diffusion_step(
        self,
        renders: Float[Tensor, "B 3 H W"],
        cameras: list[tuple[Camera, Camera]],
        df: DeepFloydGuidance,
        text_embeds,
        alpha_basis: SphericalGaussianBasis | None,
        num_prompts: int,
    ) -> Float[Tensor, ""]:
        """Evaluates the configured diffusion loss on a render batch.

        Re-noises and re-runs the UNet at every call (fresh timesteps and
        noise per invocation); `renders` may be a recomputed or a reused
        render batch, depending on the step mode.

        Args:
            renders: Render batch (B, 3, H, W), kept differentiable.
            cameras: Camera pairs (render, deformation) the renders belong
                to; deformation cameras provide the view deltas.
            df: Diffusion guidance.
            text_embeds: Pre-encoded prompt text embeddings.
            alpha_basis: The BSD prompt basis (None for 'sds' loss).
            num_prompts: Number of concept prompts.

        Returns:
            out: Diffusion loss ('loss_sds').
        """
        cfg = self.config.diffusion
        if cfg.loss == "sds":
            rt = df.SDS(renders, text_embeds, controller=None)
        else:
            with torch.no_grad():
                deltas = self.model.model.centroid - torch.stack(
                    [dc.location for rc, dc in cameras], dim=0
                )
                attn_alphas = self.get_prompt_basis(alpha_basis, deltas)
            assert renders.shape[0] * (num_prompts + 1) == text_embeds.shape[0], (
                f"render batch {renders.shape[0]} x {num_prompts + 1} branches "
                f"!= embed rows {text_embeds.shape[0]}"
            )
            rt = df.ActvnReplace(
                torch.cat([renders] * (num_prompts + 1), dim=0),
                text_embeds,
                prompt_num=num_prompts,
                controller=None,
                attn_ctrl_alphas=attn_alphas.tolist(),
            )
        return rt["loss_sds"]

    def get_training_renders(
        self, cameras: list[tuple[Camera, Camera]]
    ) -> Float[Tensor, "B 3 H W"]:
        cfg = self.config
        renders = []
        for i, (render_cam, deform_cam) in enumerate(cameras):
            light = self.randomized_light(render_cam)
            render = self.renderer(self.model.deformed(deform_cam), render_cam, light)[
                "render"
            ]
            renders.append(render)
        return torch.cat(renders, dim=0)

    def random_cam(self) -> Camera:
        cfg = self.config.camera
        radius = (
            random.random() * (self._radius_max - self._radius_min)
            + self._radius_min
        )
        F = random.random() * (cfg.F_max - cfg.F_min) + cfg.F_min
        if cfg.view_source == "random":
            return Camera.random_rot(
                H=cfg.H,
                W=cfg.W,
                F=F,
                Zn=cfg.Zn,
                Zf=cfg.Zf,
                radius=radius,
                point_upwards=cfg.point_upwards,
            ).to(self.device)
        elif cfg.view_source == "orbit":
            angle = torch.rand((), device=self.device) * 2 * torch.pi
            Q_orbit = Quaternion.from_axis_angle(
                CameraCoordinates._up.to(self.device), angle
            ).to(self.device)
            cam = Camera(
                H=cfg.H,
                W=cfg.W,
                F=F,
                Zn=cfg.Zn,
                Zf=cfg.Zf,
                co=CameraCoordinates(radius=radius, Q=Q_orbit),
            ).to(self.device)
            e_angle = (
                torch.rand((), device=self.device)
                * (cfg.elevation_max - cfg.elevation_min)
                + cfg.elevation_min
            )
            cam.rotate_from_image_space(dx=0, dy=1.0, deg=e_angle)
            return cam

    def jitter_cam(self, camera: Camera) -> Camera:
        cfg = self.config.camera
        cam = camera.copy()
        device = cam.device
        angle_range = torch.pi if cfg.aug_type == "random" else cfg.jitter_range
        angle = (torch.rand((), device=device) * 2 - 1) * angle_range
        if cfg.view_source == "random":
            axis = torch.rand((3,), device=device) - 0.5
            axis /= axis.norm().clamp(min=1e-8)
            Q_rot = Quaternion.from_axis_angle(axis, angle).to(device)
            cam.co.Q *= Q_rot
            if cfg.point_upwards:
                cam = cam.point_upwards()
        elif cfg.view_source == "orbit":
            axis = cam.co._up.to(cam.device)
            Q_rot = Quaternion.from_axis_angle(axis, angle).to(device)
            cam.co.Q *= Q_rot
        return cam

    def camera_batch(self) -> list[tuple[Camera, Camera]]:
        cfg = self.config.camera
        base_cams = [self.random_cam() for _ in range(cfg.views)]
        cameras = []
        jittered_views = cfg.aug_views
        if not cfg.force_deformation_alignment:
            jittered_views += 1
        for cam in base_cams:
            if cfg.force_deformation_alignment:
                cameras.append((cam, cam))
            for _ in range(jittered_views):
                cameras.append((self.jitter_cam(cam), cam))
        return cameras

    def log(self, epoch: int, data: dict[str, any]):
        cfg = self.config.log
        if cfg.save:
            if not hasattr(self, "_logs"):
                self._logs = {}
            self._logs[str(epoch)] = data
        if cfg.print:
            printlog = f"[Epoch {epoch}]:\n"
            for k, v in data.items():
                printlog += f"\t{k}: {v}\n"
            print(printlog)

    def finish(self):
        cfg = self.config
        run_dir = Path(cfg.path.run_dir)
        run_dir.mkdir(exist_ok=True, parents=True)
        deformation_file = run_dir / "deformation.vd3d"
        self.model.save(deformation_file)
        if cfg.log.save and hasattr(self, "_logs"):
            logs_file = run_dir / "logs.yaml"
            config_file = run_dir / "config.yaml"
            cfg.save(config_file)
            Config(self._logs).save(logs_file)
        bv = cfg.post.check_batch_views
        if bv > 0:
            batch_dir = run_dir / "batch_test"
            batch_dir.mkdir(exist_ok=True, parents=True)
            for b in range(bv):
                renders = Splimage(
                    self.get_training_renders(self.camera_batch())
                ).to_pil()
                if not isinstance(renders, list):
                    renders = [renders]
                for i, r in enumerate(renders):
                    Splimage(r).save(batch_dir / f"batch_{(i + b * bv):05}.png")

        if cfg.post.orbit:
            orbit_cfg = {
                "model": cfg.model,
                "path": {
                    "model": deformation_file,
                    "out_dir": run_dir,
                    "file_name": "orbit",
                },
            }
            orbit_script = OrbitFrames(self.device, config=Config(orbit_cfg))
            orbit_script.run()
            orbit_script.finish()
        if cfg.post.analyze:
            analyze_cfg = {
                "model": cfg.model,
                "path": {
                    "model": deformation_file,
                    "out_dir": run_dir,
                },
            }
            analyze_script = AnalyzeDeformation(self.device, config=Config(analyze_cfg))
            analyze_script.run()
            analyze_script.finish()

    def jacobian_loss(self) -> torch.Tensor:
        J_def = self.model.J_deform
        if hasattr(J_def, "weights"):
            return (self.model.J_deform.weights**2).mean()
        return (self.model.J_deform**2).mean()

    def composed_jacobian_loss(
        self, cameras: list[tuple[Camera, Camera]]
    ) -> Float[Tensor, ""]:
        """Mean squared deformation jacobian at the rendered view directions.

        Evaluates the deformation part of the composed field
        J = (I + J_delta) @ J_src at every deformation camera of the
        current batch and penalizes its squared entries. Unlike
        :meth:`jacobian_loss`, the mean is taken over composed per-view
        jacobian entries (B, F, 3, 3) instead of the num_funcs basis
        weights, so the pull per jacobian entry is independent of the
        number of basis functions (no basis dilution; directly comparable
        to a per-entry identity pull on a plain jacobian field).

        Args:
            cameras: Camera pairs (render, deformation) of this epoch.

        Returns:
            out: Scalar loss.
        """
        deltas = self.model.model.centroid - torch.stack(
            [dc.location for rc, dc in cameras], dim=0
        )
        return (self.model.delta_jacobians(deltas) ** 2).mean()

    def view_dependence_loss(self) -> Float[Tensor, ""]:
        """Per-parameter variance of the deformation weights.

        For every deformation weight (face, dim), the variance is taken
        over the N view basis functions: (F, N, D) -> var(dim=1) -> (F, D).
        Encourages the interpolated jacobian field to stay constant over
        view directions, bounding view dependence. Only valid for the
        view-dependent deformations ('vd_full', 'vd_restrained').

        Returns:
            out: Scalar loss, mean of the per-parameter variances.
        """
        return self.model.J_deform.weights.var(dim=1).mean()  # (F, N, D) -> ()

    def volume_loss(self) -> Float[Tensor, ""]:
        cams = [
            Camera.random_rot().to(self.device)
            for _ in range(self.config.loss.volume_eval_points)
        ]
        new_vols = torch.stack([self.model.deformed(cam).volume for cam in cams])
        return ((new_vols - self.model.model.volume[None]) ** 2).mean()

    def laplacian_loss(self) -> torch.Tensor:
        J = torch.einsum("bfij,cfjk->bfik", self.model.jacobians_3d(), self.model.J_src)
        V_disps = self.model.poisson.solve_poisson(J) - self.model.model.V[None]
        B, N, C = V_disps.shape
        Y = (
            torch.sparse.mm(
                self.model.model.L_cotan_csr, V_disps.permute(1, 0, 2).reshape(N, B * C)
            )
            .reshape(N, B, C)
            .permute(1, 0, 2)
        )
        return (Y**2).mean()

    def _adapt_dist(self, epoch: int):
        """Rescales the camera radius range to the current mesh bounding box.

        Port of MeshUp's `adapt_dist`: after the first epoch, the camera
        distance range is multiplied by ``max(|bbox min|, |bbox max|)`` of
        the currently deformed mesh, so the framing follows the shape as
        it deforms.

        Args:
            epoch: Current epoch index (no-op for epoch 0, matching the
                original behavior).
        """
        if epoch == 0:
            return
        with torch.no_grad():
            cam = self.random_cam()
            V = self.model.deformed(cam).V
        mult = torch.cat([V.amin(0), V.amax(0)]).abs().amax()
        self._radius_min = self.config.camera.radius_min * mult
        self._radius_max = self.config.camera.radius_max * mult

    def randomized_light(self, camera: Camera) -> LightSource:
        cfg = self.config.lighting
        align = cfg.alignment
        jitter = cfg.jitter_sigma
        device = camera.device
        strength = torch.tensor(cfg.strength, device=device)
        if align == "camera":
            origin = LightSource.from_camera(camera, strength=strength).origin
        elif align == "cosine_camera":
            # MeshUp aug_light: cosine-hemisphere sample around the camera
            # direction, placed at the camera distance. The camera direction
            # must be expressed in the rendered-mesh world frame (the same
            # swizzle/negation from_camera applies to the camera position),
            # otherwise the light lands on the opposite/behind axis.
            origin = (
                cosine_sample_hemisphere(LightSource.from_camera(camera).origin)
                * camera.location.norm()
            )
        elif align == "up":
            origin = torch.tensor(LightSource._up, device=device, dtype=torch.float32)
        else:
            raise ValueError(
                f"Alignment '{align}' is invalid for lighting alignment config."
            )
        if jitter != 0.0:
            axis = torch.rand((3,), device=device) - 0.5
            angle = torch.randn((), device=device) * jitter
            Q_rot = Quaternion.from_axis_angle(axis, angle)
            origin = Q_rot.rotate_vector(origin)
        return LightSource(origin=origin)

    def get_renderer(self) -> Renderer:
        cfg = self.config.shader
        shading = cfg.shading
        if shading == "soft":
            lambert = SoftLambertShader(apply_to={"render"}, beta=cfg.shading_beta)
        elif shading == "hard":
            lambert = HardLambertShader(apply_to={"render"})
        else:
            raise ValueError(f"Shading mode '{shading}' is invalid.")
        shaders = [
            Albedo(),
            Normal(),
            lambert,
            Alpha(apply_to={"render"}),
        ]
        if cfg.crop_fit:
            shaders.append(
                LambdaShader(
                    lambda ctx, i: ImgUtils.crop_alpha(
                        i.permute(0, 3, 1, 2),
                        border=16,
                        keep_aspect=True,
                        preserve_size=True,
                    ).permute(0, 2, 3, 1),
                    apply_to={"render"},
                )
            )
        shaders += [
            BackgroundColor(
                color=torch.tensor(cfg.bg_color, device=self.device),
                apply_to={"render"},
            ),
            Clamp(apply_to={"render"}),
        ]
        if cfg.antialias:
            shaders.append(Antialias(apply_to={"render"}))
        if cfg.resize:
            shaders.append(
                LambdaShader(
                    lambda ctx, i: resize(
                        i.permute(0, 3, 1, 2),
                        out_shape=(cfg.down_H, cfg.down_W),
                    ).permute(0, 2, 3, 1),
                    apply_to={"render"},
                )
            )
        return Renderer(shaders, self.device)
