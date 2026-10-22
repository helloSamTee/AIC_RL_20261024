"""Tests for the PPO ``Config`` dataclass and its ``validate()`` method.

This file holds two kinds of checks:

* A property-based test (Property 1) that explores the whole hyperparameter
  space: any fully in-range configuration must validate cleanly, and any
  configuration with at least one out-of-range field must be rejected with a
  ``ValueError``. Hypothesis drives this across many generated assignments.
* Example-based unit tests that pin down the documented defaults and confirm the
  full configurable hyperparameter surface required by Req 13.1 is present.

The tests import ``Config`` from the ``3_PPO`` package. Run them from the RL
workspace root so the package import resolves, e.g.::

    python -m pytest 3_PPO/tests/test_config.py -v
"""

from __future__ import annotations

import dataclasses
import importlib

from hypothesis import given, settings
from hypothesis import strategies as st

# The package directory name starts with a digit ("3_PPO"), which is not a legal
# Python identifier, so a plain ``import 3_PPO.config`` will not parse. We load
# the module through importlib instead, which accepts the dotted string name.
config_module = importlib.import_module("3_PPO.config")
Config = config_module.Config


# --- Documented valid ranges (mirrored from config.py / Req 13.3, 13.5) --------
#
# Each entry describes how to generate an in-range value and an out-of-range
# value for one field. Keeping this table in one place lets the property test
# build both "all valid" and "at least one invalid" assignments uniformly.

# Float fields: (low, high, low_inclusive, high_inclusive)
FLOAT_RANGES = {
    "learning_rate": (0.0, 1.0, False, True),   # (0, 1]
    "gamma": (0.0, 1.0, True, True),            # [0, 1]
    "gae_lambda": (0.0, 1.0, True, True),       # [0, 1]
    "clip_range": (0.0, 1.0, False, True),      # (0, 1]
    "entropy_coef": (0.0, 1.0, True, True),     # [0, 1]
    "value_coef": (0.0, 1.0, True, True),       # [0, 1]
}

# Integer fields with simple [lo, hi] ranges (minibatch_size handled specially
# because its upper bound depends on rollout_horizon).
INT_RANGES = {
    "rollout_horizon": (1, 100_000),
    "update_epochs": (1, 1_000),
    "total_steps": (1, 100_000_000),
    "eval_episodes": (1, 10_000),
}


def in_range_floats():
    """Strategy producing a dict of in-range values for every float field."""
    strategies = {}
    for name, (lo, hi, lo_inc, hi_inc) in FLOAT_RANGES.items():
        strategies[name] = st.floats(
            min_value=lo,
            max_value=hi,
            exclude_min=not lo_inc,
            exclude_max=not hi_inc,
            allow_nan=False,
            allow_infinity=False,
        )
    # max_grad_norm must be > 0; cap the upper bound to keep values sensible.
    strategies["max_grad_norm"] = st.floats(
        min_value=0.0, max_value=1_000.0, exclude_min=True,
        allow_nan=False, allow_infinity=False,
    )
    return st.fixed_dictionaries(strategies)


@st.composite
def in_range_config_kwargs(draw):
    """Draw a complete set of in-range kwargs for constructing a ``Config``.

    rollout_horizon is drawn first so minibatch_size can be constrained to
    ``[1, rollout_horizon]`` (Req 13.3), and seed is either None or an integer in
    ``[0, 2**32 - 1]`` (Req 13.5).
    """
    kwargs = draw(in_range_floats())

    for name, (lo, hi) in INT_RANGES.items():
        kwargs[name] = draw(st.integers(min_value=lo, max_value=hi))

    # minibatch_size: integer in [1, rollout_horizon].
    kwargs["minibatch_size"] = draw(
        st.integers(min_value=1, max_value=kwargs["rollout_horizon"])
    )

    # seed: None or an integer in [0, 2**32 - 1].
    kwargs["seed"] = draw(
        st.one_of(st.none(), st.integers(min_value=0, max_value=2**32 - 1))
    )
    return kwargs


# An out-of-range generator per field: each returns a value that violates that
# field's documented range. Used to inject at least one invalid field.
def out_of_range_strategy(field):
    if field in FLOAT_RANGES:
        lo, hi, lo_inc, hi_inc = FLOAT_RANGES[field]
        below = st.floats(max_value=lo, exclude_max=lo_inc,
                          allow_nan=False, allow_infinity=False)
        above = st.floats(min_value=hi, exclude_min=hi_inc,
                          allow_nan=False, allow_infinity=False)
        return st.one_of(below, above)
    if field == "max_grad_norm":
        # must be > 0, so anything <= 0 is invalid.
        return st.floats(max_value=0.0, allow_nan=False, allow_infinity=False)
    if field in INT_RANGES:
        lo, hi = INT_RANGES[field]
        return st.one_of(
            st.integers(max_value=lo - 1),
            st.integers(min_value=hi + 1),
            st.booleans(),  # bool where int is required must be rejected
        )
    if field == "minibatch_size":
        # < 1 is always invalid; the > rollout_horizon case is covered by the
        # oversize strategy below.
        return st.one_of(st.integers(max_value=0), st.booleans())
    if field == "seed":
        return st.one_of(
            st.integers(max_value=-1),
            st.integers(min_value=2**32),
            st.just(True),   # seed=True must be rejected (bool, not int)
            st.floats(allow_nan=False, allow_infinity=False),
        )
    raise AssertionError(f"no out-of-range strategy for {field}")


ALL_FIELDS = (
    list(FLOAT_RANGES)
    + ["max_grad_norm"]
    + list(INT_RANGES)
    + ["minibatch_size", "seed"]
)


# Feature: ppo-car-racing, Property 1: Configuration validation accepts in-range and rejects out-of-range
# Validates: Requirements 13.3, 13.4, 13.5, 11.3
@settings(max_examples=200)
@given(
    base=in_range_config_kwargs(),
    # Decide which fields to corrupt. An empty set => the all-in-range case.
    corrupt_fields=st.sets(st.sampled_from(ALL_FIELDS)),
    corrupt_values=st.data(),
)
def test_validation_accepts_in_range_and_rejects_out_of_range(
    base, corrupt_fields, corrupt_values
):
    if not corrupt_fields:
        # Every field is in range: validate() must return without raising, and
        # the (frozen) config values must be exactly what we supplied.
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
        # Expected. The supplied (frozen) values must remain unchanged.
        for name, value in kwargs.items():
            assert getattr(config, name) == value or (
                # NaN never equals itself; none of our generators produce NaN,
                # so a plain equality check suffices here.
                False
            )
        return
    raise AssertionError(
        f"validate() did not raise for out-of-range fields {sorted(corrupt_fields)}: "
        f"{kwargs}"
    )


# Separate, explicit case: minibatch_size larger than rollout_horizon is invalid
# (Req 13.3). This is awkward to fold into the generic corruption above because
# it is a cross-field constraint, so it gets its own property-style check.
# Feature: ppo-car-racing, Property 1: Configuration validation accepts in-range and rejects out-of-range
# Validates: Requirements 13.3
@settings(max_examples=100)
@given(
    rollout_horizon=st.integers(min_value=1, max_value=10_000),
    overflow=st.integers(min_value=1, max_value=10_000),
)
def test_minibatch_larger_than_rollout_horizon_is_rejected(rollout_horizon, overflow):
    minibatch_size = rollout_horizon + overflow
    config = Config(rollout_horizon=rollout_horizon, minibatch_size=minibatch_size)
    try:
        config.validate()
    except ValueError:
        return
    raise AssertionError(
        "validate() accepted minibatch_size "
        f"{minibatch_size} > rollout_horizon {rollout_horizon}"
    )


# --- Unit tests: config surface and defaults (Task 2.3, Req 13.1 / 13.4) -------

# The full configurable hyperparameter surface required by Req 13.1.
REQUIRED_HYPERPARAMETERS = {
    "learning_rate",
    "gamma",
    "gae_lambda",
    "rollout_horizon",
    "minibatch_size",
    "update_epochs",
    "clip_range",
    "entropy_coef",
    "value_coef",
    "max_grad_norm",
    "total_steps",
}


def test_full_hyperparameter_surface_is_present():
    """Every hyperparameter named in Req 13.1 must exist as a Config field."""
    field_names = {f.name for f in dataclasses.fields(Config)}
    missing = REQUIRED_HYPERPARAMETERS - field_names
    assert not missing, f"Config is missing required hyperparameters: {sorted(missing)}"


def test_defaults_are_applied_when_omitted():
    """Constructing Config() with no arguments applies the documented defaults."""
    config = Config()
    # Spot-check the documented defaults from the design / config.py.
    assert config.learning_rate == 3e-4
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_range == 0.2
    assert config.update_epochs == 10
    assert config.minibatch_size == 64
    assert config.rollout_horizon == 2048
    assert config.value_coef == 0.5
    assert config.entropy_coef == 0.01
    assert config.max_grad_norm == 0.5
    assert config.total_steps == 1_000_000
    assert config.eval_episodes == 5
    assert config.seed is None


def test_defaults_are_within_valid_ranges():
    """All documented defaults lie within their valid ranges (Req 13.4).

    The cleanest way to assert this is that the default Config validates cleanly.
    """
    Config().validate()  # must not raise
