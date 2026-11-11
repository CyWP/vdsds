from __future__ import annotations

import json
import uuid
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from .rendering.view_state import ViewState


class Constraint:
    """Container for constraint data as captured from the view.

    Holds a serialized ViewState snapshot plus the text prompt that was
    provided when the constraint was created or updated.
    """

    def __init__(self, view: ViewState, prompt: str = "") -> None:
        """
        Args:
            view: The captured ViewState (camera plus overlay).
            prompt: The user-provided text prompt for this constraint.
        """
        self.view = view
        self.prompt = prompt

    # ACCESS / MUTATION

    def get_view(self) -> ViewState:
        return self.view

    def set_view(self, view: ViewState) -> None:
        self.view = view

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    # SERIALIZATION

    def to_dict(self) -> dict:
        """Serialize to JSON-able types."""
        return {
            "prompt": self.prompt,
            "view": self.view.to_dict() if self.view is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Constraint:
        """Rebuild a Constraint from :meth:`to_dict` output."""
        view = data.get("view", None)
        return cls(
            view=ViewState.from_dict(view) if view is not None else None,
            prompt=data.get("prompt", ""),
        )


class ConstraintManager:
    """Manager for i-many named constraints, keyed by unique identifier.

    Constraints are stored as a dict keyed by their generated unique
    identifier (uuid4 hex string). The manager loads from its JSON file
    at construction (if the file exists) and saves on every mutation, so
    the on-disk state is always in sync with the manager.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        """
        Args:
            path: JSON file path for persistence. The file is loaded if
                it exists; with None the manager operates purely
                in-memory and becomes non-persistent.
        """
        self.path = Path(path) if path is not None else None
        # View mutations happen on the UI thread, while script threads may
        # read this dict concurrently; mutations rebuild the dict wholesale
        # so readers only ever observe complete dicts.
        self._constraints: dict[str, Constraint] = {}
        self.load()

    # ACCESS / MUTATION

    def items(self) -> dict[str, Constraint]:
        """Return a copy of the id-keyed constraint dict."""
        return dict(self._constraints)

    def get(self, cid: str) -> Constraint | None:
        """Return the constraint for an id, or None if absent."""
        return self._constraints.get(cid, None)

    def __len__(self) -> int:
        return len(self._constraints)

    def __contains__(self, cid: str) -> bool:
        return cid in self._constraints

    def add(self, view: ViewState, prompt: str = "") -> str:
        """Register a new constraint from a ViewState snapshot.

        Args:
            view: The captured ViewState (camera plus overlay).
            prompt: The user-provided text prompt.

        Returns:
            out: The generated unique id of the new constraint.
        """
        cid = uuid.uuid4().hex
        self._constraints = {
            **self._constraints,
            cid: Constraint(view=view, prompt=prompt),
        }
        self.save()
        return cid

    def remove(self, cid: str) -> None:
        """Delete the constraint with the given id."""
        if cid not in self._constraints:
            raise KeyError(cid)
        remaining = {k: v for k, v in self._constraints.items() if k != cid}
        self._constraints = remaining
        self.save()

    def set_view(self, cid: str, view: ViewState) -> None:
        """Replace a constraint's ViewState (e.g. reset it to the current view)."""
        if cid not in self._constraints:
            raise KeyError(cid)
        constraint = self._constraints[cid]
        constraint.set_view(view)
        self.save()

    def set_prompt(self, cid: str, prompt: str) -> None:
        """Replace a constraint's text prompt."""
        if cid not in self._constraints:
            raise KeyError(cid)
        self._constraints[cid].set_prompt(prompt)
        self.save()

    # SERIALIZATION

    def load(self) -> dict[str, Constraint]:
        """Load constraints from the JSON file if it exists.

        A missing or unreadable file yields an empty manager.

        Returns:
            out: The id-keyed constraint dict.
        """
        if self.path is None or not self.path.exists():
            self._constraints = {}
            return self._constraints
        try:
            with self.path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not read constraints from {self.path}: {e}")
            self._constraints = {}
            return self._constraints
        constraints = {}
        if isinstance(data, dict):
            for cid, entry in data.items():
                if isinstance(entry, dict):
                    constraints[cid] = Constraint.from_dict(entry)
        self._constraints = constraints
        return self._constraints

    def save(self, path: Path | str | None = None) -> None:
        """Write the current constraints to a JSON file.

        Args:
            path: Target path. Defaults to the manager's persistence
                path (without changing it), so passing a different path
                exports a copy while autosaves continue to the original
                file.
        """
        if self.path is None and path is None:
            return
        target = Path(path) if path is not None else self.path
        data = {cid: c.to_dict() for cid, c in self._constraints.items()}
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(target)


def prompts_from_constraints(constraints) -> dict[str, str]:
    """Extract the prompts from the constraints.

    Args:
        constraints: The id-keyed constraint dict.

    Returns:
        out: cid -> prompt for every constraint carrying a (non-empty)
            prompt. Constraints without a prompt are skipped: a
            constraint may carry only an image, only a prompt, or both.
    """
    prompts = {}
    for cid, constraint in constraints.items():
        prompt = str(constraint.prompt or "").strip()
        if prompt:
            prompts[cid] = prompt
    return prompts


def anchors_from_constraints(
    constraints,
    centroid: Float[Tensor, "3"] | None = None,
) -> dict[str, Float[Tensor, "3"]]:
    """Derive the anchor directions from the constraints' view states.

    The anchor of a constraint is the normalized direction from its
    camera to the model centroid (the basis-evaluation convention of
    the epoch batch). Only the view state matters here; the prompt is
    irrelevant.

    Args:
        constraints: The id-keyed constraint dict.
        centroid: Model centroid; defaults to the origin.

    Returns:
        out: cid -> normalized (3,) anchor for every constraint with a
            view. Constraints without a view are skipped.
    """
    centroid = (
        torch.zeros(3, dtype=torch.float32) if centroid is None else centroid
    )
    anchors = {}
    for cid, constraint in constraints.items():
        view = constraint.view if constraint is not None else None
        camera = view.camera if view is not None else None
        if camera is None:
            continue
        delta = centroid.to(camera.device) - camera.location
        anchors[cid] = delta / delta.norm().clamp_min(1e-8)
    return anchors
