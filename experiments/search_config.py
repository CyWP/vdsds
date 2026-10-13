#!/usr/bin/env python3
"""Find run folders whose config matches given key/value combinations.

Usage::

    python search_config.py SEARCH_DIR --optim.lr 0.0025 --prompt.text Moose

``SEARCH_DIR`` is searched recursively; every directory that directly
contains a config file (default ``config.yaml``, the resolved per-run config
saved by ``vdsds``) is a candidate. Filters are passed as options directly
named after the config keys, using dots for nesting (``--optim.lr``,
``--path.model``). All given filters must match (AND), and the report lists
the matching folders, one path per line.

Values are compared after normalizing both sides (e.g. ``true``/``True``
and ``3.0``/``3.00`` are equivalent). For the 2-prompt ``prompt.text`` key,
the pair is matched per individual prompt: passing one prompt matches any
run whose prompt pair contains it.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Key holding the 2-prompt pair (matches multi_gen's PROMPT_OVERRIDE_KEY).
# Its value is the pair string ``["prompt1", "prompt2"]``; it is matched per
# individual prompt rather than as a single atomic pair value.
PROMPT_KEY = "prompt.text"


def parse_dotted_options(unknown: list[str]) -> list[tuple[str, str]]:
    """Collect ``--x.y.z VALUE`` (or ``--x.y.z=VALUE``) filter pairs."""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise SystemExit(
                f"Unexpected argument '{token}'. Config filters must look like"
                " --x.y.z VALUE."
            )
        name = token[2:]
        if not name:
            raise SystemExit(f"Malformed filter option '{token}'.")
        if "=" in name:
            name, value = name.split("=", 1)
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            i += 1
            value = unknown[i]
        else:
            raise SystemExit(f"Filter option '--{name}' is missing its value.")
        pairs.append((name, value))
        i += 1
    return pairs


def parse_args() -> tuple[argparse.Namespace, list[tuple[str, str]]]:
    parser = argparse.ArgumentParser(
        description="Find run folders whose config matches key/value pairs.",
        epilog=(
            "Filters are given as direct options named after the config keys"
            " (dots for nesting), e.g. --optim.lr 0.0025; all must match (AND)."
        ),
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Directory searched recursively for run folders with configs.",
    )
    parser.add_argument(
        "--config_name",
        default="config.yaml",
        help="Config filename inside each run folder (default: config.yaml).",
    )
    args, unknown = parser.parse_known_args()
    pairs = parse_dotted_options(unknown)
    return args, pairs


def normalize(value: str) -> str:
    """Canonical string for a CLI value, matching config value reprs.

    Parsing through YAML lets ``true``/``True`` and ``3.0``/``3.00`` compare
    equal to their loaded config counterparts. Falls back to the raw string
    for values YAML cannot parse.
    """
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    return str(parsed)


def observations(key: str, value: str) -> list[str]:
    """Values representing one key/value entry for matching.

    The ``prompt.text`` pair is expanded into individual prompts; a search
    value (one prompt or a full pair string) then matches any run whose pair
    contains those prompts. All other keys match one value exactly.
    """
    if key == PROMPT_KEY:
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            prompts = [
                p.strip().strip('"').strip("'") for p in text[1:-1].split(",")
            ]
            prompts = [p for p in prompts if p]
            if prompts:
                return prompts
    return [value]


def flatten(prefix: str, value: object, out: dict[str, str]) -> None:
    """Flatten a nested mapping into ``dotted.key -> string value`` pairs."""
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten(key, v, out)
    else:
        out[prefix] = str(value)


def matches(flat: dict[str, str], wanted: dict[str, set[str]]) -> bool:
    """True if every required set of values covers what the config offers."""
    for key, required in wanted.items():
        if key not in flat:
            return False
        offered = set(observations(key, flat[key]))
        if not required.issubset(offered):
            return False
    return True


def main() -> None:
    args, filters = parse_args()
    if not args.search_dir.is_dir():
        raise SystemExit(f"Search dir '{args.search_dir}' does not exist.")
    if not filters:
        raise SystemExit("No config filters given; nothing to search for.")

    wanted: dict[str, set[str]] = {}
    for key, value in filters:
        wanted.setdefault(key, set()).update(observations(key, normalize(value)))

    matched: list[Path] = []
    for config_file in sorted(args.search_dir.rglob(args.config_name)):
        with config_file.open() as f:
            data = yaml.safe_load(f) or {}
        flat: dict[str, str] = {}
        flatten("", data, flat)
        if matches(flat, wanted):
            matched.append(config_file.parent)

    # Paths for scripting, count on stderr for humans; exit code signals hits.
    for folder in matched:
        print(folder)
    print(f"{len(matched)} run folder(s) matched.", file=sys.stderr)
    raise SystemExit(0 if matched else 1)


if __name__ == "__main__":
    main()
