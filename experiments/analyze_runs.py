#!/usr/bin/env python3
"""Analyze which config parameters influence training run outcomes.

Usage::

    python analyze_runs.py PARENT_DIR [--mode gb|rank|attrs|all] [--top N]

``PARENT_DIR`` contains outcome folders, each holding run subfolders (as
produced by ``multi_gen.py``) with a resolved ``config.yaml`` per run. All
configs are flattened to ``dotted.key`` pairs; for each parameter *value* the
report shows how often it appears in each outcome group. A value whose
frequency differs strongly between groups is a candidate cause of the
outcome. The 2-prompt pair in ``prompt.text`` is tallied per individual
prompt rather than as one value. Parameters that never vary across any run
are ignored.

Three analyses are supported, selected with ``--mode`` (default ``all`` =
every applicable one):

* ``gb``: ``good``/``bad`` subfolders (matched case-insensitively) -- binary
  correlation of parameter values with good vs bad runs.
* ``rank``: numbered subfolders (``0``, ``1``, ..., higher number = best) --
  correlation of parameter values with the numeric rank of the runs.
* ``attrs``: any remaining string-named subfolders -- each attribute's runs
  are compared against the runs of all other attribute folders (one-vs-rest,
  like good/bad), one section per attribute.

The report is printed and written to ``PARENT_DIR/<out>`` (default:
``analysis.txt``).
"""

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

# Key holding the 2-prompt pair (matches multi_gen's PROMPT_OVERRIDE_KEY).
# Its value is the pair string ``["prompt1", "prompt2"]``; it is tallied as
# individual prompts rather than as a single atomic pair value.
PROMPT_KEY = "prompt.text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate run config parameters with run outcomes."
    )
    parser.add_argument(
        "parent",
        type=Path,
        help="Folder containing the outcome subfolders (good/bad, ranked, or attributes).",
    )
    parser.add_argument(
        "--mode",
        choices=("gb", "rank", "attrs", "all"),
        default="all",
        help="Which analysis to run (default: all available ones).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report filename inside the parent dir (default: analysis.txt).",
    )
    parser.add_argument(
        "--config_name",
        default="config.yaml",
        help="Config filename inside each run folder (default: config.yaml).",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="KEY",
        help="Parameter to exclude from the analysis: an exact dotted key"
        " or a key prefix ending in a dot (e.g. path. to drop all path.*"
        " parameters). Repeatable.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only show the N most discriminative parameters per analysis "
        "(default: all).",
    )
    return parser.parse_args()


def parse_prompts(value: str) -> list[str] | None:
    """Split a ``prompt.text`` pair string into individual prompts.

    Returns ``None`` if ``value`` does not look like the pair list format
    ``["prompt1", "prompt2"]`` produced by ``multi_gen.py``.
    """
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        prompts = [
            p.strip().strip('"').strip("'") for p in text[1:-1].split(",")
        ]
        prompts = [p for p in prompts if p]
        if prompts:
            return prompts
    return None


def observations(key: str, value: str) -> list[str]:
    """Values to tally for one run: prompts individually, others as-is."""
    if key == PROMPT_KEY:
        prompts = parse_prompts(value)
        if prompts is not None:
            return prompts
    return [value]


def classify_folders(parent: Path) -> dict[str, list[Path]]:
    """Split parent's subfolders into good / bad / rank / attribute groups.

    Numeric folder names are ranks, ``good``/``bad`` (any case) are the
    binary groups, everything else is an attribute.
    """
    groups: dict[str, list[Path]] = {
        "good": [],
        "bad": [],
        "rank": [],
        "attrs": [],
    }
    for d in sorted(parent.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        name = d.name.lower()
        if name == "good":
            groups["good"].append(d)
        elif name == "bad":
            groups["bad"].append(d)
        elif d.name.isdigit():
            groups["rank"].append(d)
        else:
            groups["attrs"].append(d)
    return groups


def resolve_modes(
    parent: Path, groups: dict[str, list[Path]], mode: str
) -> list[str]:
    """Validated analyses to run, in fixed order (gb, rank, attrs)."""
    available = {
        "gb": bool(groups["good"]) and bool(groups["bad"]),
        "rank": len(groups["rank"]) >= 2,
        "attrs": len(groups["attrs"]) >= 2,
    }
    summary = ", ".join(
        f"{k}: {len(groups[k])}" for k in ("good", "bad", "rank", "attrs")
    )
    if mode == "all":
        modes = [m for m in ("gb", "rank", "attrs") if available[m]]
        if not modes:
            raise SystemExit(
                f"No analysis applicable in '{parent}'. Subfolders found "
                f"({summary}); need good+bad folders, >=2 numbered folders, "
                "or >=2 attribute folders."
            )
        return modes
    if not available.get(mode, False):
        reasons = {
            "gb": "need a 'good/' and a 'bad/' subfolder",
            "rank": "need runs in at least two numbered subfolders (0, 1, ...)",
            "attrs": "need runs in at least two non-numbered, non-good/bad subfolders",
        }
        raise SystemExit(
            f"Mode '{mode}' is not applicable in '{parent}' ({reasons[mode]}). "
            f"Subfolders found: {summary}"
        )
    return [mode]


def flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    """Flatten a nested mapping into ``dotted.key -> string value`` pairs."""
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten(key, v, out)
    else:
        out[prefix] = str(value)


def load_runs(
    folders: list[Path],
    config_name: str,
    ignore: list[str] | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Load flattened configs of every run subfolder of the given folders.

    Parameters matching an entry of ``ignore`` (exact dotted key, or prefix
    when the entry ends with a dot) are removed. Returns the list of
    flattened configs and the number of skipped subfolders (no config file
    found).
    """
    ignore = ignore or []

    def ignored(key: str) -> bool:
        return any(
            key == spec or (spec.endswith(".") and key.startswith(spec))
            for spec in ignore
        )

    configs: list[dict[str, str]] = []
    skipped = 0
    for folder in folders:
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            config_file = sub / config_name
            if not config_file.is_file():
                skipped += 1
                continue
            with config_file.open() as f:
                data = yaml.safe_load(f) or {}
            flat: dict[str, str] = {}
            flatten("", data, flat)
            configs.append({k: v for k, v in flat.items() if not ignored(k)})
    return configs, skipped


def binary_stats(
    group: list[dict[str, str]], other: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-(key, value) counts comparing a run group against all other runs.

    The score is ``freq_in_group - freq_in_other``; positive values associate
    the value with the group, negative with the other set. Parameters that
    take the same single value in every run are dropped. Returns the rows and
    the sorted list of dropped keys.
    """
    n_group, n_other = len(group), len(other)
    total = n_group + n_other
    counter_g: dict[str, Counter] = {}
    counter_o: dict[str, Counter] = {}
    raw_g: dict[str, set[str]] = {}
    raw_o: dict[str, set[str]] = {}
    keys: set[str] = set()
    runs_with_key: dict[str, int] = {}

    def tally(
        counter: dict[str, Counter],
        raw: dict[str, set[str]],
        config: dict[str, str],
    ) -> None:
        for k, v in config.items():
            runs_with_key[k] = runs_with_key.get(k, 0) + 1
            counter.setdefault(k, Counter())
            raw.setdefault(k, set()).add(v)
            for obs in observations(k, v):
                counter[k][obs] += 1

    for config in group:
        keys.update(config)
        tally(counter_g, raw_g, config)
    for config in other:
        keys.update(config)
        tally(counter_o, raw_o, config)

    rows: list[dict[str, Any]] = []
    dropped: list[str] = []
    for key in sorted(keys):
        key_g = counter_g.get(key, Counter())
        key_o = counter_o.get(key, Counter())
        values = set(key_g) | set(key_o)
        present_all = runs_with_key.get(key, 0) == total
        if key == PROMPT_KEY:
            # Constant means the exact same prompt pair in every run.
            all_pairs = raw_g.get(key, set()) | raw_o.get(key, set())
            constant = present_all and len(all_pairs) == 1
        else:
            # Constant means one single value in every run.
            constant = present_all and len(values) == 1
        if constant:
            dropped.append(key)
            continue
        for value in values:
            c_in = key_g[value]
            c_out = key_o[value]
            f_in = c_in / n_group if n_group else 0.0
            f_out = c_out / n_other if n_other else 0.0
            rows.append(
                {
                    "key": key,
                    "value": value,
                    "n_in": c_in,
                    "n_out": c_out,
                    "freq_in": f_in,
                    "freq_out": f_out,
                    "score": f_in - f_out,
                }
            )
    return rows, dropped


def pearson(xs: list[int], ys: list[int]) -> float:
    """Pearson correlation coefficient; 0.0 if either input is constant."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def rank_stats(
    ranked: list[tuple[int, dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Correlate each (key, value) with the numeric rank of its runs.

    ``ranked`` holds ``(rank, config)`` pairs where a higher rank means a
    better run. The score is the Pearson correlation between a value's
    presence and the rank (in [-1, 1]). Parameters that take the same single
    value in every run are dropped. Returns the rows and the dropped keys.
    """
    total = len(ranked)
    ranks = [rank for rank, _ in ranked]
    counter: dict[str, Counter] = {}
    raws: dict[str, set[str]] = {}
    runs_with_key: dict[str, int] = {}
    for _, config in ranked:
        for k, v in config.items():
            runs_with_key[k] = runs_with_key.get(k, 0) + 1
            counter.setdefault(k, Counter())
            raws.setdefault(k, set()).add(v)
            for obs in observations(k, v):
                counter[k][obs] += 1

    rows: list[dict[str, Any]] = []
    dropped: list[str] = []
    for key in sorted(counter):
        values = set(counter[key])
        present_all = runs_with_key[key] == total
        if key == PROMPT_KEY:
            # Constant means the exact same prompt pair in every run.
            constant = present_all and len(raws[key]) == 1
        else:
            constant = present_all and len(values) == 1
        if constant:
            dropped.append(key)
            continue
        for value in values:
            mask = [
                key in cfg and value in observations(key, str(cfg[key]))
                for _, cfg in ranked
            ]
            n = sum(mask)
            present_ranks = [r for r, m in zip(ranks, mask) if m]
            rest_ranks = [r for r, m in zip(ranks, mask) if not m]
            avg = sum(present_ranks) / n if n else float("nan")
            avg_rest = (
                sum(rest_ranks) / len(rest_ranks) if rest_ranks else float("nan")
            )
            rows.append(
                {
                    "key": key,
                    "value": value,
                    "n": n,
                    "avg": avg,
                    "avg_rest": avg_rest,
                    "score": pearson([int(m) for m in mask], ranks),
                }
            )
    return rows, dropped


def rank_keys(
    params: dict[str, list[dict[str, Any]]], top: int | None
) -> list[str]:
    """Rank parameters by how discriminative their best value is."""
    best_score = {k: max(abs(r["score"]) for r in rows) for k, rows in params.items()}
    ranked = sorted(best_score, key=best_score.get, reverse=True)
    return ranked[:top] if top is not None else ranked


def format_binary_section(
    title: str,
    rows: list[dict[str, Any]],
    dropped: list[str],
    n_in: int,
    n_out: int,
    top: int | None,
) -> str:
    """Render a binary comparison section (good/bad or attribute one-vs-rest)."""
    params: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        params.setdefault(row["key"], []).append(row)

    lines = [
        f"=== {title} ===",
        f"in: {n_in} runs   out: {n_out} runs   (score = p_in - p_out)",
        "",
    ]
    if dropped:
        lines.append(
            f"ignored {len(dropped)} constant parameter(s): {', '.join(dropped)}"
        )
        lines.append("")
    if not params:
        lines.append("no varying parameters found.")
        return "\n".join(lines)

    header = (
        f"{'value':<25} {'in':>10} {'out':>10} {'p_in':>8} {'p_out':>8} {'score':>8}"
    )
    for key in rank_keys(params, top):
        krows = sorted(
            params[key], key=lambda r: -abs(r["score"])
        )
        lines.append(f"[{key}]")
        lines.append(header)
        for r in krows:
            lines.append(
                f"{r['value']:<25} {r['n_in']:>5}/{n_in:<5} {r['n_out']:>5}/{n_out:<5}"
                f" {r['freq_in']:>7.0%} {r['freq_out']:>7.0%} {r['score']:>8.0%}"
            )
        lines.append("")
    return "\n".join(lines)


def format_rank_section(
    rows: list[dict[str, Any]],
    dropped: list[str],
    total: int,
    min_rank: int,
    max_rank: int,
    top: int | None,
) -> str:
    """Render the rank-correlation section."""
    params: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        params.setdefault(row["key"], []).append(row)

    lines = [
        "=== rank analysis ===",
        f"runs: {total}   ranks: {min_rank}..{max_rank} (higher = better)"
        "   (score = Pearson corr. of value presence with rank)",
        "avg = mean rank of runs containing the value;"
        " avg_rest = mean rank of the rest",
        "",
    ]
    if dropped:
        lines.append(
            f"ignored {len(dropped)} constant parameter(s): {', '.join(dropped)}"
        )
        lines.append("")
    if not params:
        lines.append("no varying parameters found.")
        return "\n".join(lines)

    header = f"{'value':<25} {'n':>10} {'avg':>10} {'avg_rest':>10} {'score':>8}"
    for key in rank_keys(params, top):
        krows = sorted(params[key], key=lambda r: -abs(r["score"]))
        lines.append(f"[{key}]")
        lines.append(header)
        for r in krows:
            avg = f"{r['avg']:.2f}" if r["avg"] == r["avg"] else "--"
            avg_rest = f"{r['avg_rest']:.2f}" if r["avg_rest"] == r["avg_rest"] else "--"
            lines.append(
                f"{r['value']:<25} {r['n']:>5}/{total:<5} {avg:>10}"
                f" {avg_rest:>10} {r['score']:>8.2f}"
            )
        lines.append("")
    return "\n".join(lines)


def run_gb(
    groups: dict[str, list[Path]], args: argparse.Namespace, warnings: list[str]
) -> str | None:
    """Good/bad binary analysis; warn-and-skip if it cannot run."""
    good, sk_good = load_runs(groups["good"], args.config_name, args.ignore)
    bad, sk_bad = load_runs(groups["bad"], args.config_name, args.ignore)
    if sk_good:
        warnings.append(f"{sk_good} run folder(s) under good/ had no config.")
    if sk_bad:
        warnings.append(f"{sk_bad} run folder(s) under bad/ had no config.")
    if not good or not bad:
        msg = "need at least one config in both the good and bad folders"
        if args.mode == "all":
            warnings.append(f"gb analysis skipped: {msg}")
            return None
        raise SystemExit(f"gb analysis failed: {msg}")
    rows, dropped = binary_stats(good, bad)
    return format_binary_section(
        "good/bad analysis", rows, dropped, len(good), len(bad), args.top
    )


def run_rank(
    groups: dict[str, list[Path]], args: argparse.Namespace, warnings: list[str]
) -> str | None:
    """Rank-correlation analysis; warn-and-skip if it cannot run."""
    ranked: list[tuple[int, dict[str, str]]] = []
    skipped = 0
    for folder in sorted(groups["rank"], key=lambda p: int(p.name)):
        configs, sk = load_runs([folder], args.config_name, args.ignore)
        skipped += sk
        ranked.extend((int(folder.name), cfg) for cfg in configs)
    if skipped:
        warnings.append(f"{skipped} run folder(s) under ranked folders had no config.")
    if len({rank for rank, _ in ranked}) < 2:
        msg = "need runs in at least two distinct rank folders"
        if args.mode == "all":
            warnings.append(f"rank analysis skipped: {msg}")
            return None
        raise SystemExit(f"rank analysis failed: {msg}")
    rows, dropped = rank_stats(ranked)
    return format_rank_section(
        rows,
        dropped,
        total=len(ranked),
        min_rank=min(r for r, _ in ranked),
        max_rank=max(r for r, _ in ranked),
        top=args.top,
    )


def run_attrs(
    groups: dict[str, list[Path]], args: argparse.Namespace, warnings: list[str]
) -> str | None:
    """Attribute one-vs-rest analyses; warn-and-skip for unusable attributes."""
    attrs = sorted(groups["attrs"], key=lambda p: p.name)
    per_attr: list[tuple[str, list[dict[str, str]]]] = []
    skipped = 0
    for folder in attrs:
        configs, sk = load_runs([folder], args.config_name, args.ignore)
        skipped += sk
        per_attr.append((folder.name, configs))
    if skipped:
        warnings.append(f"{skipped} run folder(s) under attribute folders had no config.")

    sections: list[str] = []
    for name, group in per_attr:
        other = [c for n, cs in per_attr if n != name for c in cs]
        if not group:
            warnings.append(f"attribute '{name}' has no runs; section skipped.")
            continue
        if not other:
            warnings.append(
                f"attribute '{name}' has no comparison group; section skipped."
            )
            continue
        rows, dropped = binary_stats(group, other)
        sections.append(
            format_binary_section(
                f"attribute '{name}'", rows, dropped, len(group), len(other), args.top
            )
        )
    return "\n\n".join(sections) if sections else None


def main() -> None:
    args = parse_args()
    parent = args.parent
    if not parent.is_dir():
        raise SystemExit(f"Parent dir '{parent}' does not exist.")
    groups = classify_folders(parent)
    modes = resolve_modes(parent, groups, args.mode)

    warnings: list[str] = []
    runners = {"gb": run_gb, "rank": run_rank, "attrs": run_attrs}
    sections: list[str] = []
    for mode in modes:
        section = runners[mode](groups, args, warnings)
        if section:
            sections.append(section)

    if not sections:
        raise SystemExit("No analysis produced a report (folder contents empty?).")

    report = "\n\n".join(sections) + "\n"
    out_path = args.out or Path("analysis.txt")
    if not out_path.is_absolute():
        out_path = parent / out_path
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"writing report to '{out_path}'")
    out_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
