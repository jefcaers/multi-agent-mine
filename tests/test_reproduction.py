"""Reproduction tests for the Madsen multi-agent mine-planning study.

Runs the fast, deterministic subset of the reconciliation gate: the
golden hand-calculation checks and the physical-invariant checks.
Every number is regenerated from fixed seeds and compared to the
values written in the paper and book. See stress_tests.py for the full
suite (bootstraps, ablations, adversarial controls, paperdiff).

Run:  pytest -q
"""
import io, os, sys, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import stress_tests as S


def _run(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn()
        except SystemExit:
            pass
    out = buf.getvalue()
    assert "[FAIL]" not in out, out
    return out


def test_golden_hand_calculations():
    out = _run(S.test_golden)
    assert "[PASS]" in out


def test_physical_invariants():
    out = _run(S.test_invariants)
    assert "[PASS]" in out
