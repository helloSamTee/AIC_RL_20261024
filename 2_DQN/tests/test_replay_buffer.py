"""Tests for the replay buffer (``replay_buffer.py``).

Covers two spec tasks:

* Task 4.2 (Property 3): capacity bound and FIFO eviction — the stored count
  never exceeds capacity; below capacity each add grows the count by one and
  retains prior transitions; at capacity each add discards exactly the oldest
  transition and keeps the count equal to capacity.
* Task 4.3 (Property 4): sampling — when the buffer holds at least ``batch_size``
  transitions, ``sample`` returns exactly ``batch_size`` distinct transitions;
  otherwise it returns ``None``.

Run from the RL workspace root, e.g.::

    python -m pytest 2_DQN/tests/test_replay_buffer.py -v
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from replay_buffer import Batch, ReplayBuffer


OBS_DIM = 3


def make_transition(tag: int):
    """Build a transition whose values encode ``tag`` so we can identify it.

    The action field stores ``tag`` directly, which lets a test check exactly
    which transitions survived eviction.
    """
    state = np.full(OBS_DIM, float(tag), dtype=np.float32)
    action = int(tag)
    reward = float(tag)
    next_state = np.full(OBS_DIM, float(tag) + 0.5, dtype=np.float32)
    terminated = bool(tag % 2)
    return state, action, reward, next_state, terminated


# Feature: dqn-lunarlander, Property 3: Replay buffer capacity bound and FIFO eviction
# Validates: Requirements 3.1, 3.2, 3.3, 3.4
@settings(max_examples=100, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=50),
    num_adds=st.integers(min_value=0, max_value=150),
)
def test_capacity_bound_and_fifo_eviction(capacity, num_adds):
    buffer = ReplayBuffer(capacity)

    for i in range(num_adds):
        count_before = len(buffer)
        buffer.add(*make_transition(i))

        # The stored count never exceeds capacity.
        assert len(buffer) <= capacity

        if count_before < capacity:
            # Below capacity: each add increases the count by exactly one.
            assert len(buffer) == count_before + 1
        else:
            # At capacity: the count stays pinned at capacity.
            assert len(buffer) == capacity

    # Final count is min(num_adds, capacity).
    assert len(buffer) == min(num_adds, capacity)

    # FIFO: the transitions still present are exactly the most recent ones.
    # Each transition's action field encodes its tag (the add index), so we can
    # read back which tags survived.
    if num_adds > 0:
        expected_surviving_tags = set(range(max(0, num_adds - capacity), num_adds))
        stored_tags = set(int(a) for a in buffer._actions[: len(buffer)])
        assert stored_tags == expected_surviving_tags


# Feature: dqn-lunarlander, Property 4: Replay buffer sampling
# Validates: Requirements 3.5, 3.6
@settings(max_examples=100, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=50),
    num_adds=st.integers(min_value=0, max_value=60),
    batch_size=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_sampling_returns_batch_or_none(capacity, num_adds, batch_size, seed):
    buffer = ReplayBuffer(capacity)
    for i in range(num_adds):
        buffer.add(*make_transition(i))

    rng = np.random.default_rng(seed)
    result = buffer.sample(batch_size, rng)

    if len(buffer) >= batch_size:
        # Enough stored: a full batch of exactly batch_size distinct transitions.
        assert isinstance(result, Batch)
        assert result.states.shape == (batch_size, OBS_DIM)
        assert result.actions.shape == (batch_size,)
        assert result.rewards.shape == (batch_size,)
        assert result.next_states.shape == (batch_size, OBS_DIM)
        assert result.terminateds.shape == (batch_size,)

        # Distinctness: because each stored transition has a unique action tag,
        # the sampled actions must have no duplicates (drawn without replacement).
        assert len(set(int(a) for a in result.actions)) == batch_size
    else:
        # Not enough stored yet: the no-minibatch indicator.
        assert result is None
