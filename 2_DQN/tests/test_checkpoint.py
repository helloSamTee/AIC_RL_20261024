"""Tests for checkpoint save/load (``checkpoint.py``).

Covers two spec tasks:

* Task 7.2 (Property 9): save/load round trip — saving a checkpoint and loading
  it into a freshly built network restores every parameter to the saved value.
* Task 7.3 (unit tests): parent-directory creation on save (8.2), a write to an
  unwritable path raises ``OSError`` (8.3), and load of a missing path raises
  ``FileNotFoundError`` naming the path (8.5).

Run from the RL workspace root, e.g.::

    python -m pytest 2_DQN/tests/test_checkpoint.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from checkpoint import load_checkpoint, save_checkpoint
from networks import QNetwork


OBS_DIM = 4
N_ACTIONS = 3


# Feature: dqn-lunarlander, Property 9: Checkpoint save/load round trip
# Validates: Requirements 8.4
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_save_load_round_trip(tmp_path_factory, seed):
    # A fresh tmp directory per example keeps Hypothesis iterations independent.
    tmp_dir = tmp_path_factory.mktemp("ckpt")
    path = os.path.join(str(tmp_dir), "model.pt")

    torch.manual_seed(seed)
    source = QNetwork(OBS_DIM, N_ACTIONS)
    # Randomize the weights so the round trip is meaningful.
    with torch.no_grad():
        for param in source.parameters():
            param.copy_(torch.randn_like(param))

    save_checkpoint(source, path)

    # Load into a freshly built network initialized differently.
    torch.manual_seed(seed + 1)
    restored = QNetwork(OBS_DIM, N_ACTIONS)
    load_checkpoint(restored, path)

    source_params = dict(source.named_parameters())
    restored_params = dict(restored.named_parameters())
    assert source_params.keys() == restored_params.keys()
    for name in source_params:
        assert torch.equal(restored_params[name], source_params[name]), (
            f"parameter {name} differs after save/load round trip"
        )


# --- Unit tests: error paths and directory creation (Task 7.3) -----------------

def test_save_creates_missing_parent_directories(tmp_path):
    """save_checkpoint creates nested parent directories if they don't exist (8.2)."""
    nested_path = os.path.join(str(tmp_path), "a", "b", "c", "model.pt")
    assert not os.path.exists(os.path.dirname(nested_path))

    network = QNetwork(OBS_DIM, N_ACTIONS)
    save_checkpoint(network, nested_path)

    assert os.path.isfile(nested_path)


def test_save_to_unwritable_path_raises_oserror(tmp_path):
    """A write whose parent path is actually a file raises OSError (8.3).

    Pointing the checkpoint "directory" at an existing regular file makes
    os.makedirs fail with an OSError, which must propagate unwrapped.
    """
    blocker = os.path.join(str(tmp_path), "not_a_dir")
    with open(blocker, "w") as handle:
        handle.write("i am a file, not a directory")

    # Ask to save underneath the file, which cannot be a directory.
    bad_path = os.path.join(blocker, "model.pt")
    network = QNetwork(OBS_DIM, N_ACTIONS)

    with pytest.raises(OSError):
        save_checkpoint(network, bad_path)


def test_load_missing_path_raises_file_not_found_naming_path(tmp_path):
    """Loading a non-existent checkpoint raises FileNotFoundError naming the path (8.5)."""
    missing_path = os.path.join(str(tmp_path), "does_not_exist.pt")
    network = QNetwork(OBS_DIM, N_ACTIONS)

    with pytest.raises(FileNotFoundError) as excinfo:
        load_checkpoint(network, missing_path)

    # The message should name the missing path so the learner can see what's wrong.
    assert "does_not_exist.pt" in str(excinfo.value)
