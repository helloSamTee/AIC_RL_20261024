"""Property-based tests for the PPO rollout buffer and GAE (tasks 5.2-5.4).

These tests exercise the pure logic in ``3_PPO/rollout_buffer.py`` with
Hypothesis. Each property runs at least 100 examples and is tagged with the
feature/property comment required by the spec. The GAE property is checked
against an INDEPENDENT plain-Python reference loop written here in the test, so
the production implementation is validated against a separate derivation rather
than against itself.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from rollout_buffer import RolloutBuffer, normalize_advantages


# ---------------------------------------------------------------------------
# Shared small-shape constants to keep the property tests fast. The buffer
# stores raw uint8 observations of shape (96, 96, 3), but the round-trip
# property does not depend on the spatial size, so we use a tiny observation
# shape and a short horizon to keep each of the 100+ examples cheap.
# ---------------------------------------------------------------------------
SMALL_OBS_SHAPE = (2, 2, 3)
MAX_HORIZON = 12
ACTION_DIM = 3


# ===========================================================================
# Task 5.2 / Property 8
# ===========================================================================
# Feature: ppo-car-racing, Property 8: Buffer storage round-trips transitions and clearing empties it
@settings(max_examples=200)
@given(
    transitions=st.lists(
        st.tuples(
            # observation: raw uint8 pixels in 0..255 of the small obs shape
            st.integers(min_value=0, max_value=255),
            # action vector components (float32-range)
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=ACTION_DIM,
                max_size=ACTION_DIM,
            ),
            # log_prob
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            # reward
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            # terminal flag
            st.booleans(),
            # value estimate
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=0,
        max_size=MAX_HORIZON,
    ),
)
def test_buffer_round_trips_and_clear_empties(transitions):
    """Every appended field is retrievable unchanged in insertion order, and
    clear() resets the stored count to zero (Req 4.2, 4.6)."""

    buffer = RolloutBuffer(
        rollout_horizon=MAX_HORIZON,
        action_dim=ACTION_DIM,
        observation_shape=SMALL_OBS_SHAPE,
    )

    # Materialize the inputs into the exact dtypes the buffer stores, so the
    # comparison is against the values as they would be stored (uint8 obs,
    # float32 action/log_prob/reward/value, bool terminal).
    expected = []
    for obs_fill, action_list, log_prob, reward, terminal, value in transitions:
        obs = np.full(SMALL_OBS_SHAPE, obs_fill, dtype=np.uint8)
        action = np.asarray(action_list, dtype=np.float32)
        buffer.add(obs, action, log_prob, reward, terminal, value)
        expected.append((obs, action, log_prob, reward, terminal, value))

    # Length reflects exactly how many transitions were appended.
    assert len(buffer) == len(expected)

    # Each stored field round-trips unchanged and in insertion order.
    for index, (obs, action, log_prob, reward, terminal, value) in enumerate(expected):
        np.testing.assert_array_equal(buffer._observations[index], obs)
        np.testing.assert_array_equal(buffer._actions[index], action)
        # Stored as float32, so compare against the float32 cast of the input.
        assert buffer._log_probs[index] == np.float32(log_prob)
        assert buffer._rewards[index] == np.float32(reward)
        assert bool(buffer._terminals[index]) == terminal
        assert buffer._values[index] == np.float32(value)

    # Clearing empties the buffer: it now holds no transitions.
    buffer.clear()
    assert len(buffer) == 0


# ===========================================================================
# Task 5.3 / Property 9
# ===========================================================================
def _reference_gae(rewards, values, terminals, gamma, gae_lambda, bootstrap_value):
    """Independent plain-Python reference implementation of GAE.

    This is deliberately a separate derivation from the production code so the
    property test validates behaviour, not a copy of the implementation. It
    follows the standard recurrence with a terminal mask:

        delta_t = reward_t + gamma * next_value * (1 - terminal_t) - value_t
        A_t     = delta_t + gamma * gae_lambda * (1 - terminal_t) * A_{t+1}

    where next_value for the final step is ``bootstrap_value`` and for earlier
    steps is the stored value of the following transition; a true terminal step
    masks both the bootstrap and the carried-back future advantage. Each return
    is ``advantage_t + value_t`` (Req 5.1-5.4).
    """

    n = len(rewards)
    advantages = [0.0] * n
    running_advantage = 0.0
    next_advantage = 0.0
    for t in range(n - 1, -1, -1):
        mask = 0.0 if terminals[t] else 1.0
        if t == n - 1:
            next_value = bootstrap_value
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        running_advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[t] = running_advantage
        next_advantage = running_advantage
    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns


# Feature: ppo-car-racing, Property 9: GAE advantages and returns match the reference, with correct masking and bootstrap
@settings(max_examples=200)
@given(
    data=st.lists(
        st.tuples(
            # reward
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            # critic value estimate
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            # terminal flag
            st.booleans(),
        ),
        min_size=1,
        max_size=MAX_HORIZON,
    ),
    gamma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    gae_lambda=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    bootstrap_value=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_gae_matches_reference(data, gamma, gae_lambda, bootstrap_value):
    """compute_gae matches an independent reference loop, with terminal masking
    and final-step bootstrap, and returns == advantages + stored values
    (Req 5.1, 5.2, 5.3, 5.4)."""

    rewards = [r for r, _, _ in data]
    values = [v for _, v, _ in data]
    terminals = [t for _, _, t in data]
    n = len(data)

    buffer = RolloutBuffer(
        rollout_horizon=n,
        action_dim=ACTION_DIM,
        observation_shape=SMALL_OBS_SHAPE,
    )
    dummy_obs = np.zeros(SMALL_OBS_SHAPE, dtype=np.uint8)
    dummy_action = np.zeros(ACTION_DIM, dtype=np.float32)
    for reward, value, terminal in zip(rewards, values, terminals):
        buffer.add(dummy_obs, dummy_action, 0.0, reward, terminal, value)

    advantages, returns = buffer.compute_gae(gamma, gae_lambda, bootstrap_value)

    # Build the reference using the SAME float32 views the buffer computes over,
    # so any difference reflects a logic difference, not a float32-vs-float64
    # rounding gap. The buffer stores rewards/values as float32.
    ref_rewards = [float(np.float32(r)) for r in rewards]
    ref_values = [float(np.float32(v)) for v in values]
    expected_adv, expected_ret = _reference_gae(
        ref_rewards, ref_values, terminals, gamma, gae_lambda, bootstrap_value
    )

    assert advantages.shape == (n,)
    assert returns.shape == (n,)
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-4, atol=1e-4)

    # Returns equal advantages plus the stored (float32) critic values.
    np.testing.assert_allclose(
        returns, advantages + np.asarray(ref_values, dtype=np.float32), rtol=1e-5, atol=1e-5
    )


# ===========================================================================
# Task 5.4 / Property 10
# ===========================================================================
# Feature: ppo-car-racing, Property 10: Advantage normalization yields zero mean and unit variance with a zero-variance guard
@settings(max_examples=200)
@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    ),
    constant_fill=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    constant_len=st.integers(min_value=2, max_value=64),
    single_value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_normalize_advantages(values, constant_fill, constant_len, single_value):
    """Normalization gives ~zero mean / ~unit variance for non-degenerate
    multi-element input, stays finite under zero variance, and leaves a
    single-element vector unchanged (Req 5.5, 5.6)."""

    arr = np.asarray(values, dtype=np.float32)

    # --- Case A: multi-element vector with non-zero variance -> mean ~0, std ~1
    if arr.size > 1 and float(arr.std()) > 1e-3:
        normalized = normalize_advantages(arr)
        assert normalized.shape == arr.shape
        assert np.all(np.isfinite(normalized))
        assert abs(float(normalized.mean())) < 1e-3
        # The epsilon in the denominator pulls std slightly below 1; it should be
        # close to 1 for a well-scaled (variance not tiny) input.
        assert abs(float(normalized.std()) - 1.0) < 1e-2

    # --- Case B: constant vector (zero variance) -> all outputs finite, no div-by-zero
    constant = np.full(constant_len, constant_fill, dtype=np.float32)
    normalized_constant = normalize_advantages(constant)
    assert normalized_constant.shape == constant.shape
    assert np.all(np.isfinite(normalized_constant))

    # --- Case C: single-element vector -> returned unchanged
    single = np.asarray([single_value], dtype=np.float32)
    normalized_single = normalize_advantages(single)
    assert normalized_single.shape == (1,)
    np.testing.assert_array_equal(normalized_single, single)
