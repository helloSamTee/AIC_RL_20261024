"""Tests for the Q-network and the target-network sync helper (``networks.py``).

Covers two spec tasks:

* Task 3.2 (Property 1): the Q-network forward pass produces output shape
  ``(batch, n_actions)``, and a target network built with the same architecture
  arguments has identical per-parameter names and shapes.
* Task 3.3 (Property 2): a hard update makes every target parameter elementwise
  equal to the corresponding source parameter.

Run from the RL workspace root, e.g.::

    python -m pytest 2_DQN/tests/test_networks.py -v
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from networks import QNetwork, hard_update


# Keep dimensions small so Hypothesis can run 100+ examples quickly.
obs_dims = st.integers(min_value=1, max_value=16)
action_counts = st.integers(min_value=1, max_value=8)
batch_sizes = st.integers(min_value=1, max_value=8)
hidden_layers = st.lists(st.integers(min_value=1, max_value=16), min_size=1, max_size=3)


# Feature: dqn-lunarlander, Property 1: Q-Network and Target-Network shapes
# Validates: Requirements 2.1, 2.2
@settings(max_examples=100, deadline=None)
@given(
    obs_dim=obs_dims,
    n_actions=action_counts,
    batch=batch_sizes,
    hidden=hidden_layers,
)
def test_qnetwork_output_shape_and_target_architecture_match(
    obs_dim, n_actions, batch, hidden
):
    hidden_tuple = tuple(hidden)
    q_network = QNetwork(obs_dim, n_actions, hidden=hidden_tuple)

    # Forward pass on a batch of observations must yield (batch, n_actions).
    observations = torch.randn(batch, obs_dim)
    output = q_network(observations)
    assert output.shape == (batch, n_actions)

    # A target network built with the same architecture arguments must have the
    # same per-parameter names and shapes as the Q-network.
    target_network = QNetwork(obs_dim, n_actions, hidden=hidden_tuple)
    q_params = dict(q_network.named_parameters())
    t_params = dict(target_network.named_parameters())
    assert q_params.keys() == t_params.keys()
    for name in q_params:
        assert q_params[name].shape == t_params[name].shape


# Feature: dqn-lunarlander, Property 2: Target-network synchronization makes parameters identical
# Validates: Requirements 2.3, 2.4
@settings(max_examples=100, deadline=None)
@given(
    obs_dim=obs_dims,
    n_actions=action_counts,
    hidden=hidden_layers,
)
def test_hard_update_makes_parameters_identical(obs_dim, n_actions, hidden):
    hidden_tuple = tuple(hidden)
    source = QNetwork(obs_dim, n_actions, hidden=hidden_tuple)
    target = QNetwork(obs_dim, n_actions, hidden=hidden_tuple)

    # Randomize so source and target almost certainly start out different.
    with torch.no_grad():
        for param in source.parameters():
            param.copy_(torch.randn_like(param))
        for param in target.parameters():
            param.copy_(torch.randn_like(param))

    hard_update(target, source)

    source_params = dict(source.named_parameters())
    target_params = dict(target.named_parameters())
    for name in source_params:
        assert torch.equal(target_params[name], source_params[name]), (
            f"parameter {name} differs after hard_update"
        )
