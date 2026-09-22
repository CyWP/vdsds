import torch

from .scripts import run_script
from .utils.config import ConfigParser


def parse_args():
    parser = ConfigParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    return parser.parse_args()


def main(args, config):
    run_script(args.task, torch.device(args.device), config)


if __name__ == "__main__":
    main(*parse_args())
