"""Tests for the TensorBoard logger (``logger.py``).

Covers two spec tasks:

* Task 8.2 (Property 8): epsilon is recorded only when it changes from the last
  logged value.
* Task 8.3 (unit tests): episode/update index sequencing starting at 0 (6.1,
  6.2), log-directory creation (6.5), and graceful degradation when creation
  fails (6.6).

These tests record the scalars actually written by capturing the underlying
``SummaryWriter.add_scalar`` calls, so they validate real logging behavior rather
than mocking it out.

Run from the RL workspace root, e.g.::

    python -m pytest 2_DQN/tests/test_logger.py -v
"""

from __future__ import annotations

import os

from hypothesis import given, settings
from hypothesis import strategies as st

import logger as logger_module
from logger import TensorBoardLogger


class _RecordingWriter:
    """A stand-in for SummaryWriter that records every add_scalar call.

    This is not a mock of behavior under test — it is a thin recorder that lets a
    test read back exactly which (tag, value, step) tuples the logger emitted.
    The logger's own change-detection and sequencing logic runs unchanged.
    """

    def __init__(self):
        self.calls = []  # list of (tag, value, step)

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


def _make_logger_with_recorder(tmp_path, monkeypatch):
    """Build a TensorBoardLogger whose writer records scalar calls."""
    recorder = _RecordingWriter()
    monkeypatch.setattr(
        logger_module, "SummaryWriter", lambda log_dir: recorder
    )
    log_dir = os.path.join(str(tmp_path), "runs")
    tb_logger = TensorBoardLogger(log_dir)
    return tb_logger, recorder


# Feature: dqn-lunarlander, Property 8: Epsilon logging records only changes
# Validates: Requirements 6.3
@settings(max_examples=100, deadline=None)
@given(
    epsilons=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    )
)
def test_epsilon_logged_only_on_change(tmp_path_factory, epsilons):
    # Build the logger with a recording writer (no monkeypatch fixture inside a
    # Hypothesis test, so patch the module attribute directly and restore it).
    recorder = _RecordingWriter()
    original_writer = logger_module.SummaryWriter
    logger_module.SummaryWriter = lambda log_dir: recorder
    try:
        tmp_dir = tmp_path_factory.mktemp("runs")
        tb_logger = TensorBoardLogger(str(tmp_dir))
        for step, epsilon in enumerate(epsilons):
            tb_logger.log_epsilon(step, epsilon)
    finally:
        logger_module.SummaryWriter = original_writer

    logged_values = [value for (tag, value, step) in recorder.calls]

    # Expected: the first value always logs; thereafter a value logs only when it
    # differs from the previously logged value (collapsing consecutive repeats).
    expected = []
    last = None
    for epsilon in epsilons:
        if last is None or epsilon != last:
            expected.append(epsilon)
            last = epsilon
    assert logged_values == expected


# --- Unit tests: index sequencing and directory handling (Task 8.3) ------------

def test_episode_and_loss_indices_pass_through_starting_at_zero(tmp_path, monkeypatch):
    """Episode-reward and loss scalars use the index the trainer supplies (6.1, 6.2)."""
    tb_logger, recorder = _make_logger_with_recorder(tmp_path, monkeypatch)

    tb_logger.log_episode_reward(0, 1.0)
    tb_logger.log_episode_reward(1, 2.0)
    tb_logger.log_loss(0, 0.5)
    tb_logger.log_loss(1, 0.25)

    episode_steps = [step for (tag, _v, step) in recorder.calls if tag == "episode/reward"]
    loss_steps = [step for (tag, _v, step) in recorder.calls if tag == "train/loss"]
    assert episode_steps == [0, 1]
    assert loss_steps == [0, 1]


def test_log_directory_is_created(tmp_path, monkeypatch):
    """The logger creates its log directory (including parents) on construction (6.5)."""
    recorder = _RecordingWriter()
    monkeypatch.setattr(logger_module, "SummaryWriter", lambda log_dir: recorder)

    nested_dir = os.path.join(str(tmp_path), "deep", "nested", "runs")
    assert not os.path.exists(nested_dir)

    TensorBoardLogger(nested_dir)
    assert os.path.isdir(nested_dir)


def test_graceful_degradation_when_directory_creation_fails(tmp_path, monkeypatch, capsys):
    """If the log directory can't be created, logging disables and never raises (6.6).

    Pointing the log dir at a path whose parent is a regular file makes
    os.makedirs raise OSError; the logger must catch it, disable itself, and turn
    subsequent log calls into no-ops so training continues.
    """
    blocker = os.path.join(str(tmp_path), "iam_a_file")
    with open(blocker, "w") as handle:
        handle.write("blocker")

    bad_log_dir = os.path.join(blocker, "runs")  # cannot create under a file

    tb_logger = TensorBoardLogger(bad_log_dir)  # must NOT raise

    assert tb_logger.logging_enabled is False
    assert tb_logger.writer is None

    # An error naming the directory is surfaced to the user.
    captured = capsys.readouterr()
    assert bad_log_dir in captured.out

    # Subsequent log calls are harmless no-ops.
    tb_logger.log_episode_reward(0, 1.0)
    tb_logger.log_loss(0, 0.5)
    tb_logger.log_epsilon(0, 0.9)
    tb_logger.log_eval_mean_reward(123.4)
    tb_logger.close()  # safe even when disabled
