"""
Unit tests for the shared async-to-sync bridge used by CrewAI tools.

These tests verify:
  - Basic coroutine execution
  - Thread-local loop reuse (no per-call overhead)
  - Cross-thread isolation (each thread gets its own loop)
  - Exception propagation
  - Recovery after a closed loop
"""

import threading

import pytest

from agents.crews.tools.async_bridge import run_async, _thread_local


# ── Helpers ──────────────────────────────────────────────────


async def _add(a: int, b: int) -> int:
    return a + b


async def _fail():
    raise ValueError("boom")


# ── Tests ────────────────────────────────────────────────────


def test_run_async_basic():
    """run_async() should return the coroutine's result."""
    result = run_async(_add(2, 3))
    assert result == 5


def test_run_async_reuses_loop():
    """Consecutive calls in the same thread should reuse the event loop."""
    _ = run_async(_add(1, 1))
    loop_a = getattr(_thread_local, "loop", None)

    _ = run_async(_add(2, 2))
    loop_b = getattr(_thread_local, "loop", None)

    assert loop_a is loop_b, "Expected the same loop object across calls"


def test_run_async_separate_threads():
    """Different threads should get different event loops and both succeed."""
    results = {}
    errors = {}

    def _worker(name: str, a: int, b: int):
        try:
            results[name] = run_async(_add(a, b))
        except Exception as exc:
            errors[name] = exc

    t1 = threading.Thread(target=_worker, args=("t1", 10, 20))
    t2 = threading.Thread(target=_worker, args=("t2", 30, 40))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, f"Thread errors: {errors}"
    assert results["t1"] == 30
    assert results["t2"] == 70


def test_run_async_exception_propagation():
    """Exceptions raised inside the coroutine should propagate to the caller."""
    with pytest.raises(ValueError, match="boom"):
        run_async(_fail())


def test_run_async_after_closed_loop():
    """If the thread-local loop is manually closed, run_async should create a new one."""
    # Run once to establish a loop
    _ = run_async(_add(1, 1))
    old_loop = getattr(_thread_local, "loop", None)
    assert old_loop is not None

    # Forcibly close it
    old_loop.close()

    # Should recover by creating a new loop
    result = run_async(_add(5, 5))
    assert result == 10

    new_loop = getattr(_thread_local, "loop", None)
    assert new_loop is not old_loop, "Should have created a fresh loop"
    assert not new_loop.is_closed()
