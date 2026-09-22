from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class Dict(dict):
    """Dictionary with recursive attribute access and nested updates."""

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if data is not None:
            self.update(data)

        if kwargs:
            self.update(kwargs)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, Dict):
            return Dict(value)

        if isinstance(value, list):
            return [Dict._wrap(v) for v in value]

        if isinstance(value, tuple):
            return tuple(Dict._wrap(v) for v in value)

        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def update(
        self,
        values: Mapping[str, Any] | None = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> None:
        """Recursively update leaves. Never replaces subdictionaries."""
        if values is None:
            values = {}

        if kwargs:
            values = {**values, **kwargs}

        self._merge(values, overwrite=overwrite)

    def _merge(
        self,
        values: Mapping[str, Any],
        *,
        overwrite: bool,
    ) -> None:
        for key, value in values.items():
            parts = key.split(".")
            self._merge_path(parts, value, overwrite=overwrite)

    def _merge_path(
        self,
        parts: list[str],
        value: Any,
        *,
        overwrite: bool,
    ) -> None:
        key = parts[0]

        if len(parts) > 1:
            if key not in self:
                self[key] = Dict()

            elif not isinstance(self[key], Dict):
                # An existing leaf cannot become a subtree.
                return

            self[key]._merge_path(
                parts[1:],
                value,
                overwrite=overwrite,
            )
            return

        # Mapping values are always merged recursively.
        if isinstance(value, Mapping):
            if key not in self:
                self[key] = Dict()

            elif not isinstance(self[key], Dict):
                # An existing leaf cannot become a subtree.
                return

            self[key]._merge(value, overwrite=overwrite)
            return

        # Scalar/leaf value.
        if key not in self or overwrite:
            self[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to ordinary Python dictionaries."""
        result = {}

        for key, value in self.items():
            if isinstance(value, Dict):
                value = value.to_dict()
            elif isinstance(value, list):
                value = [v.to_dict() if isinstance(v, Dict) else v for v in value]
            elif isinstance(value, tuple):
                value = tuple(v.to_dict() if isinstance(v, Dict) else v for v in value)

            result[key] = value

        return result


class Config(Dict):
    """Configuration dictionary loaded from YAML or JSON."""

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        path = Path(path)

        if path.suffix.lower() in {".yaml", ".yml"}:
            with path.open() as f:
                data = yaml.safe_load(f)

        elif path.suffix.lower() == ".json":
            with path.open() as f:
                data = json.load(f)

        else:
            raise ValueError(
                f"Unsupported config format: {path.suffix!r}. "
                "Expected .yaml, .yml, or .json."
            )

        if data is None:
            data = {}

        if not isinstance(data, Mapping):
            raise TypeError(
                f"Config root must be a mapping, got {type(data).__name__}."
            )

        return cls(data)

    def save(self, path: str | Path) -> None:
        """Save the configuration as JSON or YAML based on the file extension."""
        path = Path(path)
        data = self.to_dict()

        if path.suffix.lower() == ".json":
            with path.open("w") as f:
                json.dump(data, f, indent=2)

        elif path.suffix.lower() in {".yaml", ".yml"}:
            with path.open("w") as f:
                yaml.safe_dump(data, f, sort_keys=False)

        else:
            raise ValueError(
                f"Unsupported config format: {path.suffix!r}. "
                "Expected .json, .yaml, or .yml."
            )


class ConfigParser(argparse.ArgumentParser):
    """
    ArgumentParser with a YAML/JSON config and nested CLI overrides.

    Unknown arguments are interpreted as dotted config overrides:

        --model.hidden_dim 512
        --optimizer.lr 1e-4
        --training.enabled true

    Values are parsed as JSON when possible and otherwise treated
    as strings.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.add_argument(
            "--config",
            type=Path,
            required=False,
            default=None,
            help="Path to a YAML or JSON configuration file.",
        )

    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, Config]:
        args, unknown = self.parse_known_args(args, namespace)
        if args.config is None:
            config = Config({})
        else:
            config = Config.from_file(args.config)
        config.update(self._parse_overrides(unknown))

        return args, config

    @staticmethod
    def _parse_value(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @classmethod
    def _parse_overrides(
        cls,
        args: list[str],
    ) -> dict[str, Any]:
        overrides = {}

        i = 0

        while i < len(args):
            argument = args[i]

            if not argument.startswith("--"):
                raise ValueError(
                    f"Unexpected argument: {argument!r}. "
                    "Config overrides must use the form "
                    "--foo.bar value."
                )

            argument = argument[2:]

            if "=" in argument:
                key, value = argument.split("=", 1)

            else:
                key = argument

                if i + 1 >= len(args):
                    raise ValueError(f"Missing value for --{key}.")

                value = args[i + 1]
                i += 1

            if not key:
                raise ValueError("Empty config override key.")

            overrides[key] = cls._parse_value(value)
            i += 1

        return overrides
