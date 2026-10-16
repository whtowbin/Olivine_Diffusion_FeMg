"""`forster.notes`: FORSTER's warnings collected as data per thread, and issued as before
outside a `collect` block."""

import threading
import warnings
from dataclasses import replace

import numpy as np
import pytest

from forster import notes
from forster.io import fit_run
from forster.project import Project

EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"


def test_outside_collect_a_warning_is_issued_as_before():
    with pytest.warns(UserWarning, match="plain"):
        notes.warn("plain")


def test_collect_keeps_the_warning_as_data_and_issues_nothing():
    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter("always")
        with notes.collect() as caught:
            notes.warn("first")
            notes.warn("second", RuntimeWarning)
            notes.warn("first")
    assert issued == []
    assert [(n.message, n.category) for n in caught] == [
        ("first", UserWarning),
        ("second", RuntimeWarning),
        ("first", UserWarning),
    ]
    assert notes.messages(caught) == ("first", "second")


def test_held_passes_on_when_released_and_drops_otherwise():
    with notes.collect() as caught:
        with notes.held() as kept:
            notes.warn("kept")
        kept.release()
        with notes.held():
            notes.warn("dropped")
        with notes.quiet():
            notes.warn("quiet")
    assert notes.messages(caught) == ("kept",)
    with pytest.warns(UserWarning, match="released"):
        with notes.held() as kept:
            notes.warn("released")
        kept.release()


def test_threads_collect_only_their_own_warnings():
    """Two threads collecting at once, interleaved by a barrier: each list holds that
    thread's warnings only, and nothing reaches the global warnings machinery."""
    barrier = threading.Barrier(2)
    found: dict[str, tuple[str, ...]] = {}

    def work(name: str) -> None:
        with notes.collect() as caught:
            for i in range(50):
                barrier.wait()
                notes.warn(f"{name} {i}")
        found[name] = notes.messages(caught)

    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter("always")
        threads = [threading.Thread(target=work, args=(n,)) for n in ("left", "right")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert issued == []
    for name in ("left", "right"):
        assert found[name] == tuple(f"{name} {i}" for i in range(50))


@pytest.fixture(scope="module")
def short_plateau(tmp_path_factory):
    """The template example cut to end 16 um past its step: the fit warns that the
    plateau is short and that the 1-sigma range reaches the run's end (a warning held
    back while the run is extended, `notes.held`)."""
    project = Project.new(tmp_path_factory.mktemp("notes") / "p")
    run = project.run(EXAMPLE)
    return project, replace(run, initial=replace(run.initial, domain_um=(2.5, 40.0)))


def test_a_fit_collected_in_a_worker_thread_matches_the_warnings_and_result_issued(short_plateau):
    """Differential: the same fit with warnings issued (caught here on the main thread)
    and collected in a worker thread gives the same messages and identical numbers."""
    project, run = short_plateau
    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter("always")
        plain = fit_run(run, project.profiles)
    expected = tuple(dict.fromkeys(str(w.message) for w in issued))
    assert any("diffusion lengths past the last step" in m for m in expected)
    assert any("last modelled time step" in m for m in expected)

    answer = {}

    def work() -> None:
        with notes.collect() as caught:
            answer["result"] = fit_run(run, project.profiles)
        answer["messages"] = notes.messages(caught)

    with warnings.catch_warnings(record=True) as leaked:
        warnings.simplefilter("always")
        worker = threading.Thread(target=work)
        worker.start()
        worker.join()
    assert not [w for w in leaked if issubclass(w.category, UserWarning)]
    assert answer["messages"] == expected
    collected = answer["result"]
    assert collected.time_s == plain.time_s
    assert collected.indices == plain.indices
    np.testing.assert_array_equal(collected.sum_chi2, plain.sum_chi2)
    np.testing.assert_array_equal(collected.curves, plain.curves)
