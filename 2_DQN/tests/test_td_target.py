"""Tests for the temporal-difference target computation (``td_target.py``).

Covers spec task 6.2 (Property 7): for any reward, discount gamma in [0, 1], and
target-network next-state Q-values, the TD target equals ``reward`` when the
transition terminated and ``reward + gamma * max_a Q_target(next_state, a)`` when
it did not.

Run from the RL workspace root, e.g.::

    python -m pytest 2_DQN/tests/test_td_target.py -v
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from networks import QNetwork
from td_target import compute_td_target


OBS_DIM = 4
N_ACTIONS = 3


# Feature: dqn-lunarlander, Property 7: Temporal-difference target computation
# Validates: Requirements 5.4, 5.5
@settings(max_examples=100, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=8),
    gamma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    rewards_seed=st.integers(min_value=0, max_value=2**31 - 1),
    terminated_flags=st.lists(st.booleans(), min_size=1, max_size=8),
    net_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_td_target_two_cases(batch, gamma, rewards_seed, terminated_flags, net_seed):
    # Align the number of terminated flags with the batch size.
    flags = (terminated_flags * batch)[:batch]
    terminateds = torch.tensor(flags, dtype=torch.bool)

    torch.manual_seed(net_seed)
    target_network = QNetwork(OBS_DIM, N_ACTIONS)

    gen = torch.Generator().manual_seed(rewards_seed)
    rewards = torch.rand(batch, generator=gen) * 20.0 - 10.0
    next_states = torch.randn(batch, OBS_DIM, generator=gen)

    td = compute_td_target(target_network, rewards, next_states, terminateds, gamma)
    assert td.shape == (batch,)

    # Independently compute the expected target, case by case.
    with torch.no_grad():
        best_next = target_network(next_states).max(dim=1).values
    for i in range(batch):
        if flags[i]:
            # Terminal: target is just the reward (no future term).
            expected = rewards[i]
        else:
            # Non-terminal: reward + gamma * best next-state value.
            expected = rewards[i] + gamma * best_next[i]
        assert torch.isclose(td[i], expected, atol=1e-5), (
            f"index {i}: got {td[i].item()}, expected {expected.item()} "
            f"(terminated={flags[i]}, gamma={gamma})"
        )
