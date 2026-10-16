"""FORSTER's warnings, collected as data by the thread that caused them.

``warnings.catch_warnings`` swaps process-wide state (the filters and ``showwarning``),
so two threads using it at once can lose each other's warnings or leave a filter
installed for good. A GUI that fits in a worker thread while the main thread edits the
project cannot use it. FORSTER's own warnings therefore go through `warn`:

- inside a `collect` block, the warning is appended to that block's list. The block is
  held in a `contextvars.ContextVar`, so it belongs to the thread (or task) that opened
  it and the global warning state is never touched;
- outside one (scripts, the command line, tests), `warn` calls `warnings.warn` exactly
  as before, so ``pytest.warns`` and ``catch_warnings`` see the same warnings.

`held` and `quiet` are the same split for code that defers or drops its warnings.
Warnings from other libraries (numpy, scipy) still go through `warnings` as usual.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Note:
    """One collected warning: its text and category."""

    message: str
    category: type[Warning] = UserWarning

    def __str__(self) -> str:
        return self.message


_sink: ContextVar[list[Note] | None] = ContextVar("forster_notes", default=None)


def warn(message: str, category: type[Warning] = UserWarning, stacklevel: int = 1) -> None:
    """`warnings.warn`, or append to the innermost `collect` block of this thread."""
    sink = _sink.get()
    if sink is not None:
        sink.append(Note(str(message), category))
        return
    warnings.warn(message, category, stacklevel=stacklevel + 1)


@contextmanager
def collect() -> Generator[list[Note]]:
    """Collect FORSTER's warnings raised in the block (this thread only) into the list
    it yields, instead of issuing them."""
    notes: list[Note] = []
    token = _sink.set(notes)
    try:
        yield notes
    finally:
        _sink.reset(token)


def messages(notes: list[Note]) -> tuple[str, ...]:
    """The distinct messages of ``notes``, in the order first raised."""
    return tuple(dict.fromkeys(n.message for n in notes))


class Held:
    """Warnings held back by `held`; `release` issues them (or passes them on to the
    enclosing `collect` block)."""

    def __init__(self, outer: list[Note] | None) -> None:
        self.outer = outer
        self.caught: list = []

    def release(self) -> None:
        if self.outer is not None:
            self.outer.extend(self.caught)
            return
        for w in self.caught:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


@contextmanager
def held() -> Generator[Held]:
    """Hold back the block's warnings; call ``release()`` on the yielded object after
    the block to issue them, or drop them by not calling it."""
    outer = _sink.get()
    hold = Held(outer)
    if outer is None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hold.caught = caught
            yield hold
        return
    token = _sink.set(hold.caught)
    try:
        yield hold
    finally:
        _sink.reset(token)


@contextmanager
def quiet() -> Generator[None]:
    """Drop the block's warnings."""
    if _sink.get() is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
        return
    token = _sink.set([])
    try:
        yield
    finally:
        _sink.reset(token)
