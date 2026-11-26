import random, torch
from pathlib import Path
from vdsds.scripts.train_sds import TrainModelSDS
from vdsds.utils.config import Config
from vdsds.utils.deepfloyd import DeepFloydGuidance
from vdsds.rendering import Renderer, LightSource

dev = torch.device("cuda:0")
random.seed(0); torch.manual_seed(0)

config = Config()
config.update(TrainModelSDS._default_config_overrides)
config.update(
    {
        "model": {"name": "mesh"},
        "prompt": {"view_dependent": True},
        "path": {
            "model": str(Path.home() / "3D/models/vdsds/cow_subdiv.glb"),
            "constraints": "cow_dds_constraints_v2.json",
        },
        "optim": {"epochs": 1},
        "loss": {"constraint_sds_bank": 4, "constraint_sds_draws": 2},
    }
)
script = TrainModelSDS(dev, config)
script._apply_profile()
script.constraints = script.get_constraints()
script._prune_constraint_losses()
df = DeepFloydGuidance(config.diffusion, dev)
script._df = df
prompts, text_embeds, prompt_basis = script.get_text_embeds(df)
script.viewset = text_embeds if hasattr(text_embeds, "view_weights") else None
script.text_embeds = text_embeds
script.prompt_basis = prompt_basis
script._embed_batch = config.camera.views
script._radius_min = config.camera.radius_min
script._radius_max = config.camera.radius_max
script.model.train(model=False, weights=True)

img_cons = [
    (cid, c) for cid, c in script.constraints.items().items() if c.type == "image"
]
renderer = Renderer.basic(bg_color=torch.ones(3, device=dev), device=dev, background=False)

@torch.no_grad()
def eval_loss():
    vals = []
    for cid, c in img_cons:
        for slot in range(len(script._dds_bank)):
            t, noise = script._dds_bank[slot]
            vals.append(
                float(
                    script.constraint_sds_loss_for(cid, t, noise, eval_mode=True)
                )
            )
    return sum(vals) / len(vals)

@torch.no_grad()
def eval_ious():
    out = []
    for cid, c in img_cons:
        cam = c.view.camera.copy().to(dev)
        ref = c.view.overlay.image(cam.H, cam.W).to(dev)[0, 3] > 0.5
        r = renderer(script.model.deformed(cam), cam, LightSource.from_camera(cam))["render"]
        out.append(round(float(((r[0, 3] > 0.5) & ref).float().sum() / ((r[0, 3] > 0.5) | ref).float().sum().clamp_min(1)), 3))
    return out

print("eval loss:", round(eval_loss(), 4), "| IoUs:", eval_ious())
opt = torch.optim.Adam(script.model.get_parameters(), lr=0.0025)
for step in range(100):
    opt.zero_grad()
    loss = script.constraint_sds_loss(draws=2)
    loss.backward()
    opt.step()
    if step % 25 == 24:
        print(f"step {step}: eval loss {eval_loss():.4f} IoUs {eval_ious()}")
