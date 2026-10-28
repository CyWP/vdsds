#!/usr/bin/env python3
"""Continually run randomized vdsds training experiments.

Each experiment samples one value per key from the ``--overrides`` YAML file
(one value per key, ``python -m vdsds`` is launched in a fresh subprocess with
those overrides). Two exceptions: the ``config`` key (one value forwarded as
the ``--config`` base config) and the ``prompts`` key: ``--num_prompts``
prompts (default 2) are sampled and passed via ``--prompt.text``.

Each run writes its outputs to ``<run_dir>/<Prompt1_Prompt2>``, where spaces
and punctuation in the prompts are replaced by underscores. If the same
combination (name) comes up again, a number suffix is added. A candidate model
(.glb file) is randomly sampled from ``models_dir`` and passed via
``--path.model``.
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

# Where the vdsds package lives (not an installed package).
REPO_ROOT = SCRIPT_DIR.parent

# YAML key holding the prompt pool; sampled as an ``--num_prompts``-long
# tuple, and surfaced as the ``--prompt.text`` override.
PROMPTS_KEY = "prompts"
PROMPT_OVERRIDE_KEY = "prompt.text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run randomized vdsds training experiments."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Directory under which per-experiment subfolders are created.",
    )
    parser.add_argument(
        "--num_runs",
        nargs="?",
        type=int,
        default=1,
        help="Number of experiments to run.",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Directory of candidate .glb models; one is sampled per run.",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        required=True,
        help="YAML file mapping override keys to lists of candidate values.",
    )
    parser.add_argument(
        "--task",
        default="train_sds",
        help="vdsds task name (default: train_sds).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for vdsds (default: cuda:0).",
    )
    parser.add_argument(
        "--duplicate_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the same prompt to be sampled twice (default: true); "
        "pass --no-duplicate_prompts to force two distinct prompts.",
    )
    parser.add_argument(
        "--force_duplicate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Always sample a single prompt and duplicate it in both prompt "
        "slots (default: false). Takes precedence over --duplicate_prompts.",
    )
    parser.add_argument(
        "--model_pattern",
        default="*.glb",
        help="Glob pattern for candidate models inside --models_dir (default: *.glb).",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        choices=(1, 2),
        default=2,
        help="Number of prompts sampled per run (default 2). With 1, the "
        "override is a 1-element list (1 basis function / 1 anchor).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the sampled configurations without running them.",
    )
    return parser.parse_args()


def slugify(prompt: str) -> str:
    """Replace spaces and punctuation with underscores."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_")
    return slug or "prompt"


def sample_overrides(
    overrides: dict[str, list],
    allow_duplicates: bool = True,
    force_duplicate: bool = False,
    num_prompts: int = 2,
) -> dict[str, str]:
    """Randomly pick a value for each override key.

    The ``prompts`` key is the exception: ``num_prompts`` prompts are sampled
    and combined under the ``--prompt.text`` override. ``force_duplicate``
    picks a single prompt and repeats it as a pair (2-prompt runs only);
    otherwise the prompt count comes from ``num_prompts`` and
    ``allow_duplicates`` decides whether the two may be the same one.
    """
    sampled = {}
    for key, values in overrides.items():
        if key == PROMPTS_KEY:
            continue
        sampled[key] = str(random.choice(values))

    pool = overrides[PROMPTS_KEY]
    if num_prompts == 1:
        sampled[PROMPT_OVERRIDE_KEY] = f'["{random.choice(pool)}"]'
    elif force_duplicate:
        prompt1 = prompt2 = random.choice(pool)
    elif allow_duplicates:
        prompt1, prompt2 = random.choice(pool), random.choice(pool)
    else:
        prompt1, prompt2 = random.sample(pool, 2)
    if num_prompts != 1:
        sampled[PROMPT_OVERRIDE_KEY] = f'["{prompt1}", "{prompt2}"]'
    return sampled


def build_command(
    sampled: dict[str, str], task: str, run_path: Path, model_path: Path, device: str
) -> list[str]:
    """Full argv for ``python -m vdsds`` with all sampled overrides."""
    command = [sys.executable, "-m", "vdsds", "--task", task, "--device", device]
    for key, value in sampled.items():
        command.extend([f"--{key}", value])
    command.extend(["--path.run_dir", str(run_path)])
    command.extend(["--path.model", str(model_path)])
    return command


def launch(command: list[str]) -> None:
    """Run one experiment in a fresh subprocess, importable via PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    python_path = (
        str(REPO_ROOT) if existing is None else os.pathsep.join((str(REPO_ROOT), existing))
    )
    env["PYTHONPATH"] = python_path
    subprocess.run(command, check=True, env=env, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    with args.overrides.open() as f:
        overrides = yaml.safe_load(f)

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the overrides file in the experiment root for reference
    # (individual runs save their own resolved configs in their subfolders).
    snapshot = run_dir / args.overrides.name
    if args.overrides != snapshot:
        shutil.copy2(args.overrides, snapshot)

    # Track (slug -> count) so repeated prompt combinations get numbered names.
    seen: dict[str, int] = {}

    for i in range(args.num_runs):
        sampled = sample_overrides(
            overrides,
            allow_duplicates=args.duplicate_prompts,
            force_duplicate=args.force_duplicate,
            num_prompts=args.num_prompts,
        )
        run_name = slugify_run_name(run_dir, sampled, seen)

        run_path = run_dir / run_name
        model_path = sample_model(args.models_dir, args.model_pattern)
        command = build_command(
            sampled, args.task, run_path, model_path, args.device
        )

        print(f"=== Experiment {i + 1}/{args.num_runs}: {run_path} ===")
        print(" ".join(command))
        if args.dry_run:
            continue

        try:
            launch(command)
        except subprocess.CalledProcessError as e:
            print(f"Run {i + 1} failed with exit code {e.returncode}; continuing.")
        except KeyboardInterrupt:
            print("Interrupted; stopping after current run.")
            break



def sample_model(models_dir: Path, pattern: str) -> Path:
    """Randomly choose one of the candidate model files in ``models_dir``."""
    models = sorted(models_dir.glob(pattern))
    if not models:
        raise FileNotFoundError(f"No '{pattern}' files found in '{models_dir}'.")
    return random.choice(models)


def slugify_run_name(run_dir: Path, sampled: dict[str, str], seen: dict[str, int]) -> str:
    """Build a unique run folder name from the sampled prompts."""
    prompts = [
        p.strip().strip('"')
        for p in sampled[PROMPT_OVERRIDE_KEY][1:-1].split(",")
    ]
    slug = "_".join(slugify(p) for p in prompts)

    count = seen.get(slug, 0)
    run_name = slug if count == 0 else f"{slug}_{count + 1}"
    while (run_dir / run_name).exists():
        count += 1
        run_name = slug if count == 0 else f"{slug}_{count + 1}"
    seen[slug] = count + 1
    return run_name


if __name__ == "__main__":
    main()
