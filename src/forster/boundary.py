"""Boundary histories: instantaneous growth stages and rim values that change in time.

Coordinates are rim-relative, as measured: x = 0 is the crystal edge at the end of
the run. Stage 0 is the starting crystal of an `InitialCondition`, with its rim at
``domain_um[0]`` held at ``edge_fo``. Each later `Stage` starts at ``start_s``
(seconds after stage 0 began) and

- moves the rim outward to ``rim_um`` (at or below the previous rim), filling the
  new zone between the two rims with ``zone_value``; with ``rim_um`` equal to the
  previous rim it only changes the rim value (a melt that changes composition);
- holds ``rim_value`` (default ``zone_value``) at the rim from then on.

The model grid covers the final crystal; nodes outside the crystal at a given time
are inactive. Shared by the explicit steppers (`forster.model`) and, later, the
implicit scheme of session github-drafts-68. Plan:
``docs/plans/2026-10-13-growth-stages.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forster.model import InitialCondition


@dataclass(frozen=True)
class Stage:
    """An instantaneous growth step, or a change of rim value, at ``start_s``."""

    start_s: float
    rim_um: float
    rim_value: float | None = None
    zone_value: float | None = None

    @property
    def held_value(self) -> float:
        """The value held at the rim during this stage."""
        value = self.rim_value if self.rim_value is not None else self.zone_value
        if value is None:
            raise ValueError("a stage needs rim_value, zone_value or both")
        return value


@dataclass(frozen=True)
class BoundaryHistory:
    """Stage 0 (the starting crystal) followed by the later stages, in time order."""

    stages: tuple[Stage, ...]

    @classmethod
    def from_initial(cls, initial: InitialCondition, stages: Sequence[Stage]) -> BoundaryHistory:
        first = Stage(0.0, initial.domain_um[0], rim_value=initial.edge_fo)
        history = cls((first, *stages))
        history.check(fraction=True)
        return history

    def check(self, *, fraction: bool = False) -> None:
        """Start times increasing, rims never moving inward, every new zone given a value.

        ``fraction``: values are Fo fractions and must lie in (0, 1].
        """
        for before, stage in zip(self.stages, self.stages[1:], strict=False):
            if not stage.start_s > before.start_s:
                raise ValueError(
                    f"stage start times must increase: {stage.start_s} s after {before.start_s} s"
                )
            if stage.rim_um > before.rim_um:
                raise ValueError(
                    f"the rim cannot move inward: {stage.rim_um} um after {before.rim_um} um"
                )
            if stage.rim_um < before.rim_um and stage.zone_value is None:
                raise ValueError(
                    f"the stage at {stage.start_s} s grows the crystal from {before.rim_um} to "
                    f"{stage.rim_um} um: give zone_value, the composition of the new zone"
                )
            values = [stage.held_value] + (
                [stage.zone_value] if stage.zone_value is not None else []
            )
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"stage values must be finite, got {values}")
            if fraction and not all(0.0 < v <= 1.0 for v in values):
                raise ValueError(f"Fo values must be fractions (0-1); got {values}")

    @property
    def outer_rim_um(self) -> float:
        return min(s.rim_um for s in self.stages)

    def first_nodes(self, model_x_um: np.ndarray) -> list[int]:
        """Index of each stage's rim on the model grid (the nearest node, as `model_grid`
        places the domain ends)."""
        return [int(np.abs(model_x_um - s.rim_um).argmin()) for s in self.stages]

    def start_rows(self, dt_s: float) -> list[int]:
        """Time step at which each stage begins (nearest step)."""
        return [round(s.start_s / dt_s) for s in self.stages]
