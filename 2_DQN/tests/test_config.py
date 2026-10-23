"""Tests for the DQN ``Config`` dataclass and its ``validate()`` method.

Covers two spec tasks:

* Task 2.2 (Property 11): configuration validation rejects out-of-range values.
  A fully in-range config validates cleanly; a config with at least one
  out-of-range field is rejected with a ``ValueError``.
* Task 2.3 (unit tests): the documented defaults are applied when values are
  omitted, and the full configurable hyperparameter surface is present.

Run from the RL workspace root, e.g.::

    python -m pytest 2_DQN/tests/test_config.py -v
"""

from __future__ import annotations

import dataclasses

from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config


# --- Documented valid ranges (mirrored from config.py / Req 10.3, 10.5) --------
#
# Float fields: (low, high, low_inclusive, high_inclusive)
FLOAT_RANGES = {
    "learning_rate": (0.0, 1.0, False, True),   # (0, 1]
    "gamma": (0.0, 1.0, True, True),            # [0, 1]
    "epsilon_start": (0.0, 1.0, True, True),    # [0, 1]
}

# Integer fields with simple [lo, hi] ranges. Cross-field constraints
# (batch_size <= buffer_capacity, learning_starts <= buffer_capacity,
# epsilon_min <= epsilon_start) are handled by using a fixed, generous
# buffer_capacity / epsilon_start so the generic generators stay in range.
INT_RANGES = {
    "buffer_capacity": (1, 1_000_000),
    "target_update_interval": (1, 100_000),
    "epsilon_decay_steps": (1, 100_000_000),
    "total_steps": (1, 100_000_000),
    "updates_per_step": (1, 1_000),
    "eval_episodes": (1, 10_000),
}


@st.composite
def in_range_config_kwargs(draw):
    """Draw a complete set of in-range kwargs for constructing a ``Config``.

    buffer_capacity is drawn first so batch_size and learning_starts can be
    constrained to ``[1, buffer_capacity]``, and epsilon_start is drawn so
    epsilon_min can be constrained to ``[0, epsilon_start]``.
    """
    kwargs = {}
    for name, (lo, hi, lo_inc, hi_inc) in FLOAT_RANGES.items():
        kwargs[name] = draw(
            st.floats(
                min_value=lo,
                max_value=hi,
                exclude_min=not lo_inc,
                exclude_max=not hi_inc,
                allow_nan=False,
                allow_infinity=False,
            )
        )

    for name, (lo, hi) in INT_RANGES.items():
        kwargs[name] = draw(st.integers(min_value=lo, max_value=hi))

    capacity = kwargs["buffer_capacity"]
    kwargs["batch_size"] = draw(st.integers(min_value=1, max_value=capacity))
    kwargs["learning_starts"] = draw(st.integers(min_value=1, max_value=capacity))

    # epsilon_min: in [0, epsilon_start].
    kwargs["epsilon_min"] = draw(
        st.floats(
            min_value=0.0,
            max_value=kwargs["epsilon_start"],
            allow_nan=False,
            allow_infinity=False,
        )
    )

    # seed: None or a non-negative integer.
    kwargs["seed"] = draw(
        st.one_of(st.none(), st.integers(min_value=0, max_value=2**32 - 1))
    )
    return kwargs


def out_of_range_strategy(field):
    """Return a strategy that produces an out-of-range value for ``field``."""
    if field in FLOAT_RANGES:
        lo, hi, lo_inc, hi_inc = FLOAT_RANGES[field]
        below = st.floats(max_value=lo, exclude_max=lo_inc,
                          allow_nan=False, allow_infinity=False)
        above = st.floats(min_value=hi, exclude_min=hi_inc,
                          allow_nan=False, allow_infinity=False)
        return st.one_of(below, above)
    if field in INT_RANGES:
        lo, hi = INT_RANGES[field]
        return st.one_of(
            st.integers(max_value=lo - 1),
            st.integers(min_value=hi + 1),
            st.booleans(),  # bool where int is required must be rejected
        )
    if field in ("batch_size", "learning_starts"):
        # < 1 is always invalid regardless of buffer_capacity.
        return st.one_of(st.integers(max_value=0), st.booleans())
    if field == "epsilon_min":
        # < 0 or > 1 is always invalid. (> epsilon_start is covered separately.)
        return st.one_of(
            st.floats(max_value=0.0, exclude_max=True,
                      allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0, exclude_min=True,
                      allow_nan=False, allow_infinity=False),
        )
    if field == "seed":
        return st.one_of(
            st.integers(max_value=-1),
            st.just(True),  # seed=True must be rejected (bool, not int)
        )
    raise AssertionError(f"no out-of-range strategy for {field}")


ALL_FIELDS = (
    list(FLOAT_RANGES)
    + list(INT_RANGES)
    + ["batch_size", "learning_starts", "epsilon_min", "seed"]
)


# Feature: dqn-lunarlander, Property 11: Configuration validation rejects out-of-range values
# Validates: Requirements 2.5, 9.3, 10.3, 10.5
@settings(max_examples=200)
@given(
    base=in_range_config_kwargs(),
    corrupt_fields=st.sets(st.sampled_from(ALL_FIELDS)),
    corrupt_values=st.data(),
)
def test_validation_accepts_in_range_and_rejects_out_of_range(
    base, corrupt_fields, corrupt_values
):
    if not corrupt_fields:
        # Every field is in range: validate() must return without raising.
        config = Config(**base)
        config.validate()  # should not raise
        for name, value in base.items():
            assert getattr(config, name) == value
        return

    # At least one field is out of range: validate() must raise ValueError.
    kwargs = dict(base)
    for field in corrupt_fields:
        kwargs[field] = corrupt_values.draw(
            out_of_range_strategy(field), label=f"bad_{field}"
        )

    config = Config(**kwargs)
    try:
        config.validate()
    except ValueError:
        return  # expected
    raise AssertionError(
        f"validate() did not raise for out-of-range fields {sorted(corrupt_fields)}: "
        f"{kwargs}"
    )


# Separate explicit cross-field cases that are awkward to fold into the generic
# corruption above because they depend on another field's value.
# Feature: dqn-lunarlander, Property 11: Configuration validation rejects out-of-range values
# Validates: Requirements 10.3
@settings(max_examples=100)
@given(
    buffer_capacity=st.integers(min_value=1, max_value=10_000),
    overflow=st.integers(min_value=1, max_value=10_000),
)
def test_batch_size_larger_than_buffer_capacity_is_rejected(buffer_capacity, overflow):
    batch_size = buffer_capacity + overflow
    config = Config(buffer_capacity=buffer_capacity, batch_size=batch_size)
    try:
        config.validate()
    except ValueError:
        return
    raise AssertionError(
        f"validate() accepted batch_size {batch_size} > buffer_capacity {buffer_capacity}"
    )


# Feature: dqn-lunarlander, Property 11: Configuration validation rejects out-of-range values
# Validates: Requirements 10.3
@settings(max_examples=100)
@given(
    epsilon_start=st.floats(min_value=0.0, max_value=1.0,
                            allow_nan=False, allow_infinity=False),
    bump=st.floats(min_value=0.001, max_value=1.0,
                   allow_nan=False, allow_infinity=False),
)
def test_epsilon_min_above_epsilon_start_is_rejected(epsilon_start, bump):
    epsilon_min = epsilon_start + bump
    # Only meaningful when epsilon_min stays within [0, 1] so the failure is the
    # cross-field check, not the plain range check.
    if epsilon_min > 1.0:
        epsilon_min = 1.0
    if epsilon_min <= epsilon_start:
        return  # not an interesting case
    config = Config(epsilon_start=epsilon_start, epsilon_min=epsilon_min)
    try:
        config.validate()
    except ValueError:
        return
    raise AssertionError(
        f"validate() accepted epsilon_min {epsilon_min} > epsilon_start {epsilon_start}"
    )


# --- Unit tests: config surface and defaults (Task 2.3, Req 10.1 / 10.4) -------

# The full configurable hyperparameter surface required by Req 10.1.
REQUIRED_HYPERPARAMETERS = {
    "learning_rate",
    "gamma",
    "buffer_capacity",
    "batch_size",
    "learning_starts",
    "target_update_interval",
    "epsilon_start",
    "epsilon_min",
    "epsilon_decay_steps",
    "total_steps",
    "updates_per_step",
    "eval_episodes",
    "seed",
    "log_dir",
    "video_dir",
    "checkpoint_path",
}


def test_full_hyperparameter_surface_is_present():
    """Every hyperparameter named in Req 10.1 must exist as a Config field."""
    field_names = {f.name for f in dataclasses.fields(Config)}
    missing = REQUIRED_HYPERPARAMETERS - field_names
    assert not missing, f"Config is missing required hyperparameters: {sorted(missing)}"


def test_defaults_are_applied_when_omitted():
    """Constructing Config() with no arguments applies the documented defaults."""
    config = Config()
    assert config.learning_rate == 1e-3
    assert config.gamma == 0.99
    assert config.buffer_capacity == 100_000
    assert config.batch_size == 64
    assert config.learning_starts == 1_000
    assert config.target_update_interval == 1_000
    assert config.epsilon_start == 1.0
    assert config.epsilon_min == 0.05
    assert config.epsilon_decay_steps == 50_000
    assert config.total_steps == 500_000
    assert config.updates_per_step == 1
    assert config.eval_episodes == 10
    assert config.seed is None
    assert config.log_dir == "artifacts/runs/dqn-lunarlander"
    assert config.video_dir == "artifacts/videos/dqn-lunarlander"
    assert config.checkpoint_path == "artifacts/checkpoints/dqn-lunarlander.pt"


def test_defaults_are_within_valid_ranges():
    """All documented defaults lie within their valid ranges (Req 10.4)."""
    Config().validate()  # must not raise
