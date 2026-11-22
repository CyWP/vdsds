# https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/deep_floyd_guidance.py

import torch
import torch.nn.functional as F
from diffusers import IFPipeline
from jaxtyping import Float
from torch import nn
from torch import Tensor

from .ptp import (
    register_activation_control,
    save_forward,
    unregister_attention_control,
)


def perpneg_combine(
    eps_branches: torch.Tensor,
    eps_uncond: torch.Tensor,
    pos_idx: torch.Tensor,
    neg_weights: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Perp-Neg score-space combination (Armandpour et al., 2023).

    Combines per-branch noise predictions into one guidance direction per
    render: the uncond score as base, the positive branch's direction at
    the guidance scale, and every other branch's direction with its
    component parallel to the positive direction removed (the part that
    would damage the shared content), scaled by the negative weights.

    Args:
        eps_branches: Per-branch, per-render noise predictions, RGB
            channels only, (V, B, ...).
        eps_uncond: Uncond noise predictions, (B, ...).
        pos_idx: Positive branch index per render, (B,).
        neg_weights: Negative weights per (render, branch), (B, V); the
            positive branch's weight must be 0.
        guidance_scale: CFG scale applied to the positive direction.

    Returns:
        out: Combined noise prediction per render, (B, ...).
    """
    d = eps_branches - eps_uncond[None]  # (V, B, ...)
    rows = torch.arange(eps_branches.shape[1], device=d.device)
    d0 = d[pos_idx, rows]  # (B, ...)
    # Orthogonalize every branch direction against the positive direction,
    # flattened per render (the whole score tensor is the vector space).
    shape = d.shape
    x = d.reshape(*shape[:2], -1)  # (V, B, C)
    y = d0.reshape(shape[1], -1)  # (B, C)
    y2 = y.pow(2).sum(-1, keepdim=True).clamp_min(1e-8)  # (B, 1)
    xy = (x * y[None]).sum(-1, keepdim=True)  # (V, B, 1)
    perp = (x - (xy / y2[None]) * y[None]).reshape(shape)  # (V, B, ...)
    neg = neg_weights.permute(1, 0)  # (V, B)
    # Broadcast the per-render weights over the score tensor's trailing dims.
    neg = neg.reshape(neg.shape + (1,) * (d.dim() - 2))  # (V, B, 1, ..., 1)
    return (
        eps_uncond
        + guidance_scale * d0
        + (neg * perp).sum(dim=0)
    )


class DeepFloydGuidance(nn.Module):
    def __init__(self, cfg, device="cuda", t_range=[0.02, 0.98]):
        super().__init__()
        # Create model
        # torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = device
        if cfg.model_size == "XL":
            model_name = "DeepFloyd/IF-I-XL-v1.0"
        elif cfg.model_size == "L":
            model_name = "DeepFloyd/IF-I-L-v1.0"
        elif cfg.model_size == "M":
            model_name = "DeepFloyd/IF-I-M-v1.0"
        if cfg.dtype == "float32":
            dtype = torch.float32
            variant = "fp32"
        elif cfg.dtype == "float16":
            dtype = torch.float16
            variant = "fp16"
        self.model_name = model_name
        self.variant = variant
        self.dtype = dtype
        self.pipe = IFPipeline.from_pretrained(
            model_name,
            variant=variant,
            torch_dtype=dtype,
            # text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        self.weights_dtype = dtype
        # self.pipe.enable_xformers_memory_efficient_attention()
        if cfg.cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        # self.pipe.enable_attention_slicing(1)
        self.pipe.unet.to(memory_format=torch.channels_last)
        # p.requires_grad_(False)
        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def encode_text(self, prompt, negative_prompt=None, batch_size=1):
        pos_embeds, neg_embeds = self.pipe.encode_prompt(
            prompt, negative_prompt=negative_prompt
        )
        pos_embeds = torch.cat([pos_embeds] * batch_size, dim=0)
        neg_embeds = torch.cat([neg_embeds] * batch_size, dim=0)
        prompt = torch.cat([pos_embeds, neg_embeds])
        return prompt

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def encode_text_2(
        self, prompt, negative_prompt=None, batch_size=1, return_mask=False
    ):
        new_prompt = []
        max_length = 77
        for text in prompt:
            new_prompt.extend([text] * batch_size)
        if return_mask:
            mask = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )["attention_mask"]
        pos_embeds, neg_embeds = self.pipe.encode_prompt(
            new_prompt, negative_prompt=negative_prompt
        )
        pos_embeds = torch.cat([pos_embeds], dim=0)
        neg_embeds = torch.cat([neg_embeds], dim=0)[
            : (pos_embeds.shape[0] // len(prompt))
        ]
        prompt = torch.cat([pos_embeds, neg_embeds])
        if return_mask:
            return prompt, mask
        return prompt

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(self, latents, t, text_embeds):
        input_dtype = latents.dtype
        return self.pipe.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            text_embeds.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def SDS(
        self,
        rgb,
        text_embeds,
        rgb_as_latents=False,
        grad_embeds=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        # rgb_BCHW = rgb.permute(0, 3, 1, 2)
        assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
        rgb = rgb * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False)
        # we can potentially do this later
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            int(self.max_step + 1),
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        # add noise
        noise = torch.randn_like(latents)  # TODO: use torch generator
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        # predict the noise residual with unet, NO grad! (unless the text
        # embeddings themselves are being fitted)
        if grad_embeds:
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                text_embeds,
            )  # (B, 6, 64, 64)
        else:
            with torch.no_grad():
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    text_embeds,
                )  # (B, 6, 64, 64)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # Keep the graph to text_embeds when fitting them (grad_embeds);
        # otherwise detach so only the latents/rgb receive gradients.
        target = latents - grad if grad_embeds else (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "grad": grad,
        }

    def DDS(
        self,
        rgb: Float[Tensor, "B 3 H W"],
        ref_rgb: Float[Tensor, "B 3 H W"],
        text_embed: Float[Tensor, "T D"] | Float[Tensor, "B T D"],
        rgb_as_latents: bool = False,
        t: Float[Tensor, "B"] | None = None,
        noise: Float[Tensor, "B 4 64 64"] | None = None,
        t_max: float = 0.5,
    ) -> dict[str, Float[Tensor, ""] | Float[Tensor, "B 3 64 64"] | float]:
        """Paired-noise score-difference loss (Delta Denoising Score
        with a fixed reference image).

        The render and the reference are noised with the SAME
        (timestep, noise) draw and both cond-branch scores are
        evaluated; the gradient is their difference. Everything the two
        UNet calls share (the noise anchor, the unconditional offset,
        the stochastic directions inherited from the shared draw)
        cancels, so the loss is a relative signal "render toward
        reference" without ever forming the CFG combination: the
        cond-only difference is O(1e-2) in fp16 and the CFG scale is
        left to the caller's loss weight.

        The reference branch carries no gradient (the reference image
        is fixed), so a draw costs ONE grad-enabled UNet pass over a
        two-row batch. The timestep band defaults to the lower half of
        the schedule: at high noise both scores collapse toward the
        unconditional field and the difference loses its signal.

        Args:
            rgb: Render batch (B, 3, H, W) in [0, 1], kept
                differentiable.
            ref_rgb: Reference batch (B, 3, H, W) in [0, 1], same
                framing/resolution as `rgb` (already cropped to the
                shared content window by the caller); detached inside.
            text_embed: The shared conditioning embedding, (T, D) (one
                per-batch row) or (B, T, D).
            rgb_as_latents: Unsupported (same as :meth:`SDS`).
            t: Optional fixed timesteps (B,); sampled in the lower
                `t_max` band of the schedule when None.
            noise: Optional fixed noise (B, 4, 64, 64); sampled fresh
                when None. Passing both enables fully reproducible
                draws (tests).
            t_max: Upper fraction of the noise schedule to sample t
                from (0 < t_max <= 1); only used when `t` is None.

        Returns:
            out: {'loss_sds', 'grad_norm', 'grad'} like :meth:`SDS`.
        """
        assert rgb_as_latents is False, f"No latent space in {self.__class__.__name__}"
        assert rgb.shape == ref_rgb.shape, (
            f"Render/reference shape mismatch: {tuple(rgb.shape)} vs "
            f"{tuple(ref_rgb.shape)}."
        )
        batch_size = rgb.shape[0]
        rgb = rgb * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        ref_rgb = ref_rgb * 2.0 - 1.0
        latents = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False)
        ref_latents = F.interpolate(ref_rgb, (64, 64), mode="bilinear", align_corners=False)
        if t is None:
            # Lower band of the schedule: the paired difference needs
            # conditional signal, which vanishes toward pure noise.
            hi = int(self.min_step + t_max * (self.max_step - self.min_step))
            t = torch.randint(
                self.min_step, max(hi, self.min_step + 1), [batch_size],
                dtype=torch.long, device=self.device,
            )
        if noise is None:
            noise = torch.randn(
                (batch_size, *latents.shape[1:]), device=self.device
            )
        # Same (t, noise) draw for both branches: the pairing.
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        ref_noisy = self.scheduler.add_noise(ref_latents, noise, t)
        if text_embed.ndim == 2:
            text_embed = text_embed[None]
        text_embed = text_embed.expand(batch_size, -1, -1)
        model_input = torch.cat([latents_noisy, ref_noisy], dim=0)
        embeds = torch.cat([text_embed, text_embed], dim=0)
        with torch.no_grad():
            eps = self.forward_unet(model_input, torch.cat([t] * 2), embeds)
        # Keep only the RGB channels (IF also predicts a variance head).
        eps_render, eps_ref = eps[:batch_size, :3], eps[batch_size:, :3]
        grad = (eps_render - eps_ref) * (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "grad": grad,
        }

    def PerpNegSDS(
        self,
        rgb,
        branch_embeds,
        uncond_embeds,
        pos_idx,
        neg_weights,
        rgb_as_latents=False,
    ):
        """SDS loss with Perp-Neg view negatives (score-space).

        Runs one UNet pass per view branch (plus uncond) over the render
        batch, then combines the branches with :func:`perpneg_combine`:
        the branch selected by `pos_idx` guides at the CFG scale while
        the other views push orthogonally to it with `neg_weights`.

        Args:
            rgb: Render batch (B, 3, H, W), kept differentiable.
            branch_embeds: Conditioning embedding per view branch per
                render, (V, B, T, D).
            uncond_embeds: The uncond embedding, (T, D).
            pos_idx: Positive branch index per render, (B,).
            neg_weights: Negative weights per (render, branch), (B, V),
                0 at the positive branch.

        Returns:
            out: {'loss_sds', 'grad_norm', 'grad'} like :meth:`SDS`.
        """
        batch_size = rgb.shape[0]
        num_views = branch_embeds.shape[0]
        assert rgb_as_latents is False, f"No latent space in {self.__class__.__name__}"
        rgb = rgb * 2.0 - 1.0
        latents = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False)
        t = torch.randint(
            self.min_step,
            int(self.max_step + 1),
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # One latent copy per view branch, then the uncond copy.
        model_input = torch.cat([latents_noisy] * num_views + [latents_noisy], dim=0)
        text_embeds = torch.cat(
            [
                branch_embeds.reshape(num_views * batch_size, *branch_embeds.shape[2:]),
                uncond_embeds[None].expand(batch_size, -1, -1),
            ],
            dim=0,
        )
        with torch.no_grad():
            eps = self.forward_unet(
                model_input, torch.cat([t] * (num_views + 1)), text_embeds
            )
        # Keep only the RGB channels (IF also predicts a variance head).
        eps_branches = eps[: num_views * batch_size, :3].reshape(
            num_views, batch_size, 3, *eps.shape[2:]
        )
        eps_uncond = eps[num_views * batch_size :, :3]
        eps_hat = perpneg_combine(
            eps_branches, eps_uncond, pos_idx, neg_weights, self.cfg.guidance_scale
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (eps_hat - noise)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "grad": grad,
        }

    def ActvnReplace(
        self,
        rgb,
        text_embeds,
        iter=None,
        modified_cfg=False,
        prompt_num=2,
        attn_ctrl_alphas=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0] // (prompt_num + 1)
        if attn_ctrl_alphas is not None:
            alpha_t = torch.as_tensor(attn_ctrl_alphas)
            if alpha_t.dim() == 1:
                if alpha_t.shape[0] != prompt_num - 1:
                    raise ValueError(
                        "attn_ctrl_alphas as [n_prompts-1] requires length "
                        f"{prompt_num - 1}, got {alpha_t.shape[0]}"
                    )
            elif alpha_t.dim() == 2:
                if alpha_t.shape[1] != prompt_num - 1 or alpha_t.shape[0] not in (
                    1,
                    batch_size,
                ):
                    raise ValueError(
                        "attn_ctrl_alphas as [B, n_prompts-1] requires shape "
                        f"[{batch_size}, {prompt_num - 1}] (or [1, {prompt_num - 1}]), "
                        f"got {tuple(alpha_t.shape)}"
                    )
            else:
                raise ValueError(
                    "attn_ctrl_alphas must be [n_prompts-1], [1, n_prompts-1], "
                    f"or [B, n_prompts-1]; got ndim {alpha_t.dim()}"
                )
        rgb = rgb * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False)
        # we can potentially do this later
        t_mask = torch.randint(
            int(self.min_step),
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        t_mask = torch.cat(
            [t_mask] * prompt_num
        )  # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t_ = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        t = torch.cat([t_] * prompt_num)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # we simply cache the previous forward pass function so we can recover
            save_forward(self.pipe)
            register_activation_control(
                self.pipe, None, prompt_num=prompt_num, replace_alpha=attn_ctrl_alphas
            )
            noise_ = torch.randn_like(
                latents[: (latents.shape[0] // (prompt_num + 1))]
            )  # TODO: use torch generator
            noise = torch.cat([noise_] * prompt_num)
            latents_noisy = self.scheduler.add_noise(
                latents[: batch_size * prompt_num], noise, t
            )
            latents_noisy_uncond = self.scheduler.add_noise(
                latents[batch_size * prompt_num :], noise_, t_
            )
            t_noise_pred_text = self.forward_unet(
                latents_noisy,
                t,
                text_embeds[: batch_size * prompt_num],
            )
            # restore the original attention forward cached at the top of
            # this call so the uncond pass runs clean (unpatched) attention
            unregister_attention_control(self.pipe)
            t_noise_pred_uncond = self.forward_unet(
                latents_noisy_uncond,
                t_,
                text_embeds[batch_size * prompt_num :],
            )  # (B, 6, 64, 64)
        t_noise_pred_text = t_noise_pred_text.chunk(prompt_num)[-1]
        t_noise_pred_text, t_predicted_variance = t_noise_pred_text.split(3, dim=1)
        t_noise_pred_uncond, _ = t_noise_pred_uncond.split(3, dim=1)
        t_noise_pred = t_noise_pred_text + self.cfg.guidance_scale * (
            t_noise_pred_text - t_noise_pred_uncond
        )
        w = (1 - self.alphas[t_]).view(-1, 1, 1, 1)
        grad = w * (t_noise_pred - noise_)  # * mask[:8]
        grad = torch.nan_to_num(grad)

        latents = latents.chunk(prompt_num + 1)[-2]
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "grad": grad,
        }
