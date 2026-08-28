import argparse
import json
from pathlib import Path

import torch
import yaml

from .scripts.view_model import ViewModel
from .scripts.train import TrainModel


def load_config(path: str) -> dict:
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    return args, config


def main(args, config):
    task = args.task
    model_path = args.model
    device = torch.device(args.device)
    if task == "view":
        ViewModel(model_path=model_path, device=device).launch()
    elif task == "train":
        TrainModel(model_path=model_path, device=device, config=config).launch()
    else:
        raise ValueError(f"Task '{task}' is unrecognized.")


if __name__ == "__main__":
    args, config = parse_args()
    main(args, config)
