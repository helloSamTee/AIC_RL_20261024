"""Property-based tests for the continuous-action policy (``3_PPO/policy.py``).

These cover three universal correctness properties from the PPO CarRacing spec:

- Property 5: produced actions always lie within the environment bounds.
- Property 6: the tanh-corrected log-probability is always finite.
- Property 7: the evaluation action is deterministic and noise-free.

Each test drives the real implementation with ``torch`` tensors (the type the
policy functions expect) and uses Hypothesis to explore the input space,
including the numerically delicate near-saturation region of ``tanh`` where a
naive log-prob would blow up to ``-inf``.
"""

from __future__ import annotations

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from policy import (
    clamp_to_bounds,
    deterministic_action,
    tanh_log_prob,
    tanh_rescale,
)

# Use float64 throughout so the tests check the mathematics of the
# implementation rather than float32 rounding noise.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Hypothesis strategies: build valid, well-formed inputs for the policy math.
# ---------------------------------------------------------------------------

# Action dimensionality. CarRacing uses 3, but the math is per-dimension, so we
# exercise a small range of sizes to make sure the component-wise handling holds.
action_dims = st.integers(min_value=1, max_value=5)

# Finite real values. We allow large magnitudes (up to ~20) so tanh is driven
# deep into saturation, which is exactly where the stable log-prob identity
# matters.
finite_reals = st.floats(
    min_value=-20.0,
    max_value=20.0,
    allow_nan=False,
    allow_infinity=False,
)

# Strictly positive standard deviations (the Actor guarantees std > 0 via
# exp(log_std)). Kept away from 0 to stay a valid Gaussian scale.
positive_std = st.floats(
    min_value=1e-3,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
)


def _vector(draw, dim, elements):
    """Draw a length-``dim`` float64 tensor from the given element strategy."""
    values = draw(st.lists(elements, min_size=dim, max_size=dim))
    return torch.tensor(values, dtype=DTYPE)


@st.composite
def pre_squash_and_bounds(draw):
    """A pre-squash action vector together with valid per-dimension bounds.

    Returns ``(raw_action, low, high)`` where ``low < high`` component-wise.
    """
    dim = draw(action_dims)
    raw_action = _vector(draw, dim, finite_reals)
    low = _vector(draw, dim, finite_reals)
    # Build ``high`` as ``low + strictly-positive width`` so low < high always.
    widths = _vector(
        draw,
        dim,
        st.floats(min_value=1e-3, max_value=40.0, allow_nan=False, allow_infinity=False),
    )
    high = low + widths
    return raw_action, low, high


@st.composite
def gaussian_and_sample(draw):
    """A Gaussian ``(mean, std)`` plus a pre-squash sample ``raw_action``.

    The sample is drawn to include near-saturation magnitudes so the tanh
    correction term is exercised in its hardest regime.
    """
    dim = draw(action_dims)
    mean = _vector(draw, dim, finite_reals)
    std = _vector(draw, dim, positive_std)
    raw_action = _vector(draw, dim, finite_reals)
    return mean, std, raw_action


# ---------------------------------------------------------------------------
# Property 5: Produced actions always lie within the environment bounds.
# ---------------------------------------------------------------------------
# Feature: ppo-car-racing, Property 5: Produced actions always lie within the environment bounds
# Validates: Requirements 3.2, 3.5
@settings(max_examples=200)
@given(pre_squash_and_bounds())
def test_actions_lie_within_bounds(data):
    raw_action, low, high = data

    # The real action pipeline: squash+rescale, then the floating-point clamp.
    rescaled = tanh_rescale(raw_action, low, high)
    action = clamp_to_bounds(rescaled, low, high)

    # Every component must sit inside its inclusive per-dimension band, even for
    # the extreme pre-squash magnitudes Hypothesis throws at it.
    assert torch.all(action >= low), f"component below low: {action} < {low}"
    assert torch.all(action <= high), f"component above high: {action} > {high}"
    assert torch.all(torch.isfinite(action)), f"non-finite action: {action}"


# ---------------------------------------------------------------------------
# Property 6: The tanh-corrected log-probability is always finite.
# ---------------------------------------------------------------------------
# Feature: ppo-car-racing, Property 6: The tanh-corrected log-probability is always finite
# Validates: Requirements 3.3
@settings(max_examples=200)
@given(gaussian_and_sample())
def test_tanh_log_prob_is_finite(data):
    mean, std, raw_action = data

    log_prob = tanh_log_prob(mean, std, raw_action)

    # The whole point of the stable identity: no NaN, no +/-inf, even when the
    # sample drives tanh deep into saturation.
    assert torch.all(torch.isfinite(log_prob)), (
        f"non-finite log_prob {log_prob} for mean={mean}, std={std}, raw={raw_action}"
    )


# ---------------------------------------------------------------------------
# Property 7: The evaluation action is deterministic and noise-free.
# ---------------------------------------------------------------------------
# Feature: ppo-car-racing, Property 7: The evaluation action is deterministic and noise-free
# Validates: Requirements 3.4, 11.1
@settings(max_examples=200)
@given(pre_squash_and_bounds())
def test_deterministic_action_is_noise_free_and_repeatable(data):
    mean, low, high = data

    action = deterministic_action(mean, low, high)

    # 1. It equals tanh(mean) rescaled onto the bounds (no sampled noise): the
    #    affine map from [-1, 1] applied to tanh(mean).
    expected = low + (torch.tanh(mean) + 1.0) * 0.5 * (high - low)
    assert torch.allclose(action, expected, atol=1e-12), (
        f"deterministic action {action} != expected tanh-rescale {expected}"
    )

    # 2. It lies within the bounds.
    assert torch.all(action >= low)
    assert torch.all(action <= high)

    # 3. It is identical across repeated calls with the same inputs.
    again = deterministic_action(mean, low, high)
    assert torch.equal(action, again), "deterministic action changed between calls"
