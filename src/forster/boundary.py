"""Boundary histories: instantaneous growth stages and rim values that change in time.

Coordinates are rim-relative, as measured: x = 0 is the crystal edge at the end of
the run. Stage 0 is the starting crystal of an `InitialCondition`, with its rim at
``domain_um[0]`` held at ``edge_fo``. Each later `Stage` starts at ``start_s``
(seconds after stage 0 began) and

- moves the rim outward to ``rim_um`` (at or below the previous rim), filling the
  new zone between the two rims with ``zone_value``; with ``rim_um`` equal to the
  previous rim it only changes the rim value (a melt that changes composition);
- holds ``rim_value`` (default ``zone_value``) at the rim from then on.

A rim value is a number (fixed) or changes in time: a `RimPath` (piecewise linear
through given times) or any function of the time in seconds since stage 0 began.
Changing values need the implicit scheme (`forster.implicit`).

The model grid covers the final crystal; nodes outside the crystal at a given time
are inactive. Shared by the explicit steppers (`forster.model`) and, later, the
implicit scheme of session github-drafts-68. Plan:
``docs/plans/2026-10-13-growth-stages.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forster.model import InitialCondition


SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class RimPath:
    """A rim value changing in time: linear between ``(times_days[i], values[i])``,
    ``values[0]`` before the first time and ``values[-1]`` after the last.

    Times are days since stage 0 began. Called with a time in seconds.
    """

    times_days: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times_days) != len(self.values) or not self.values:
            raise ValueError("RimPath needs one value per time, and at least one")
        if any(b <= a for a, b in zip(self.times_days, self.times_days[1:], strict=False)):
            raise ValueError(f"RimPath times must increase, got {self.times_days}")

    def __call__(self, t_s: float) -> float:
        return float(np.interp(t_s / SECONDS_PER_DAY, self.times_days, self.values))

    @property
    def last_change_s(self) -> float:
        return self.times_days[-1] * SECONDS_PER_DAY


Rim = float | Callable[[float], float]


def is_fixed(rim: Rim) -> bool:
    return isinstance(rim, int | float)


def rim_at(rim: Rim, t_s: float) -> float:
    """The rim value at ``t_s`` seconds after stage 0 began."""
    return float(rim) if isinstance(rim, int | float) else float(rim(t_s))


def settles_at_s(rim: Rim) -> float | None:
    """Time after which ``rim`` no longer changes: 0 for a fixed value, the last time
    of a `RimPath`, None (unknown) for any other function."""
    if is_fixed(rim):
        return 0.0
    return rim.last_change_s if isinstance(rim, RimPath) else None


@dataclass(frozen=True)
class Stage:
    """An instantaneous growth step, or a change of rim value, at ``start_s``."""

    start_s: float
    rim_um: float
    rim_value: Rim | None = None
    zone_value: float | None = None

    @property
    def rim(self) -> Rim:
        """The value held at the rim during this stage, fixed or changing."""
        value = self.rim_value if self.rim_value is not None else self.zone_value
        if value is None:
            raise ValueError("a stage needs rim_value, zone_value or both")
        return value

    @property
    def held_value(self) -> float:
        """The fixed value held at the rim during this stage (explicit schemes)."""
        rim = self.rim
        if not isinstance(rim, int | float):
            raise ValueError(
                f"the stage at {self.start_s} s has a changing rim value; that needs "
                'scheme="implicit"'
            )
        return float(rim)


@dataclass(frozen=True)
class BoundaryHistory:
    """Stage 0 (the starting crystal) followed by the later stages, in time order."""

    stages: tuple[Stage, ...]

    @classmethod
    def from_initial(
        cls, initial: InitialCondition, stages: Sequence[Stage], rim: Rim | None = None
    ) -> BoundaryHistory:
        """Stage 0 holds ``rim`` (default ``initial.edge_fo``) at ``domain_um[0]``."""
        first = Stage(0.0, initial.domain_um[0], rim_value=initial.edge_fo if rim is None else rim)
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
            rim = stage.rim
            values = [rim_at(rim, stage.start_s)] + (
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
