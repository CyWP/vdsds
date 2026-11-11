import torch

from .scripts import run_script
from .utils.config import ConfigParser


def parse_args(args: list[str] | None = None):
    parser = ConfigParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument(
        "--constraints",
        type=str,
        required=False,
        default=None,
        help="Path to a constraints JSON file to load (and autosave to).",
    )
    args, config = parser.parse_args(list(args) if args is not None else None)
    # Constraints are in-memory by default; only an explicit flag loads a
    # JSON file.
    if args.constraints is not None:
        config.update({"path.constraints": args.constraints})
    return args, config


def main(args, config):
    run_script(args.task, torch.device(args.device), config)


if __name__ == "__main__":
    main(*parse_args())
