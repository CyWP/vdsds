import argparse
import torch

from .scripts.view_model import ViewModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    return parser.parse_args()


def main(args):
    task = args.task
    model_path = args.model
    device = torch.device(args.device)
    if task == "view":
        ViewModel(model_path=model_path, device=device).launch()
    else:
        raise ValueError(f"Task '{task}' is unrecognized.")


if __name__ == "__main__":
    main(parse_args())
