"""Tests for the epsilon-greedy policy (``policy.py``).

Covers two spec tasks:

* Task 5.2 (Property 5): action-selection validity and greedy tie-breaking — the
  selected action index is always within ``[0, n_actions)``; greedy selection
  returns the lowest index attaining the maximum Q-value.
* Task 5.3 (Property 6): epsilon schedule bounds and monotonicity — scheduled
  epsilon always lies within ``[epsilon_min, epsilon_start]`` and is
  monotonically non-increasing in the step.

Run from the RL workspace root, e.g.::

    python -m pytest 2_DQN/tests/test_policy.py -v
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config
from policy import epsilon_at, select_action


q_value_lists = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=10,
)
epsilons = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
seeds = st.integers(min_value=0, max_value=2**32 - 1)


# Feature: dqn-lunarlander, Property 5: Action-selection validity and greedy tie-breaking
# Validates: Requirements 4.1, 4.2, 9.1
@settings(max_examples=100, deadline=None)
@given(q_values=q_value_lists, epsilon=epsilons, seed=seeds, greedy=st.booleans())
def test_action_validity_and_greedy_tie_breaking(q_values, epsilon, seed, greedy):
    q_array = np.array(q_values, dtype=np.float64)
    n_actions = len(q_array)
    rng = np.random.default_rng(seed)

    action = select_action(q_array, epsilon, n_actions, rng, greedy=greedy)

    # Always a valid index.
    assert isinstance(action, int)
    assert 0 <= action < n_actions

    # Greedy selection (greedy=True, or epsilon=0 which forces exploit) must
    # return the lowest index attaining the maximum Q-value.
    expected_greedy = int(np.argmax(q_array))  # numpy argmax breaks ties to lowest index
    if greedy:
        assert action == expected_greedy
    elif epsilon == 0.0:
        assert action == expected_greedy


# Feature: dqn-lunarlander, Property 5: Action-selection validity and greedy tie-breaking
# Validates: Requirements 4.2, 9.1
@settings(max_examples=100, deadline=None)
@given(
    n_actions=st.integers(min_value=1, max_value=10),
    tie_value=st.floats(min_value=-10.0, max_value=10.0,
                        allow_nan=False, allow_infinity=False),
    seed=seeds,
)
def test_greedy_breaks_ties_toward_lowest_index(n_actions, tie_value, seed):
    # Every action shares the same (maximum) Q-value, so the only correct greedy
    # choice under lowest-index tie-breaking is action 0.
    q_array = np.full(n_actions, tie_value, dtype=np.float64)
    rng = np.random.default_rng(seed)
    action = select_action(q_array, epsilon=0.0, n_actions=n_actions, rng=rng, greedy=True)
    assert action == 0


# Feature: dqn-lunarlander, Property 6: Epsilon schedule bounds and monotonicity
# Validates: Requirements 4.3, 4.4
@settings(max_examples=100, deadline=None)
@given(
    epsilon_start=st.floats(min_value=0.0, max_value=1.0,
                            allow_nan=False, allow_infinity=False),
    epsilon_min_frac=st.floats(min_value=0.0, max_value=1.0,
                               allow_nan=False, allow_infinity=False),
    epsilon_decay_steps=st.integers(min_value=1, max_value=1_000_000),
    steps=st.lists(st.integers(min_value=-1000, max_value=2_000_000),
                   min_size=2, max_size=20),
)
def test_epsilon_bounds_and_monotonicity(
    epsilon_start, epsilon_min_frac, epsilon_decay_steps, steps
):
    # Derive epsilon_min <= epsilon_start so the Config is internally consistent.
    epsilon_min = epsilon_min_frac * epsilon_start
    cfg = Config(
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay_steps=epsilon_decay_steps,
    )

    # Bounds: every scheduled epsilon lies within [epsilon_min, epsilon_start].
    for step in steps:
        value = epsilon_at(step, cfg)
        assert epsilon_min - 1e-9 <= value <= epsilon_start + 1e-9

    # Monotonic non-increasing: sorting the steps ascending, epsilon never rises.
    ascending_steps = sorted(steps)
    previous = epsilon_at(ascending_steps[0], cfg)
    for step in ascending_steps[1:]:
        current = epsilon_at(step, cfg)
        assert current <= previous + 1e-9
        previous = current
