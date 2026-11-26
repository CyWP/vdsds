#!/bin/bash
micromamba run -n 3dml python -m vdsds --task train_sds \
  --path.model "$HOME/3D/models/vdsds/cow_subdiv.glb" \
  --model.name mesh --prompt.view_dependent true \
  --constraints cow_dds_constraints_v3.json \
  \
  --optim.epochs 400 --optim.seed 0 \
  --path.run_dir outputs/dds_run6_phased \
  --post.checkpoint_every 100 \
  --schedule '[
    {"epochs": 150, "profile": "none", "loss.diffusion": 0.0,
     "loss.composed_jacobian": 0.0, "loss.view_dependence": 0.0,
     "loss.silhouette": 1000, "loss.silhouette_inverse": 1.0,
     "loss.depth": 10.0},
    {"epochs": 250, "profile": "none", "loss.diffusion": 1.0,
     "loss.composed_jacobian": 500.0, "loss.view_dependence": 3000.0,
     "loss.silhouette": 1000, "loss.silhouette_inverse": 1.0,
     "loss.depth": 10.0}
  ]' > /tmp/opencode/dds_run6.log 2>&1
