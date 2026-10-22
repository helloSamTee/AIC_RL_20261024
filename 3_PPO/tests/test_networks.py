"""Tests for the CNN encoder, Actor, and Critic networks (``3_PPO/networks.py``).

This module holds:

- Property 2 (Hypothesis): pixel scaling maps 0..255 into 0.0..1.0 (Req 2.6).
- Property 3 (Hypothesis): the encoder produces a fixed-length feature vector (Req 2.1).
- Property 4 (Hypothesis): the Actor standard deviation is strictly positive (Req 2.3).
- Unit tests: encoder shape-mismatch rejection (Req 2.2) and Critic scalar output (Req 2.4).

The CNNEncoder expects a channels-first batch of observations with shape
``(N, 3, 96, 96)``, so every input constructed here follows that layout. Batch
sizes are kept small so the convolutions stay fast under Hypothesis.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from networks import (
    OBS_CHANNELS,
    OBS_HEIGHT,
    OBS_WIDTH,
    ActorHead,
    CNNEncoder,
    CriticHead,
    scale_pixels,
)


# ---------------------------------------------------------------------------
# Property 2 — Pixel scaling maps 0..255 into 0.0..1.0
# ---------------------------------------------------------------------------
# Feature: ppo-car-racing, Property 2: Pixel scaling maps 0..255 into 0.0..1.0
@settings(max_examples=150)
@given(
    # A small 4-D batch of integer pixels in the valid 0..255 range. Shapes are
    # kept tiny because the property is purely arithmetic and shape-independent.
    data=st.lists(
        st.integers(min_value=0, max_value=255),
        min_size=1,
        max_size=64,
    ),
)
def test_property_2_pixel_scaling_maps_0_255_into_unit_interval(data: list[int]) -> None:
    raw = torch.tensor(data, dtype=torch.uint8)

    scaled = scale_pixels(raw)

    # Output is floating point.
    assert scaled.dtype == torch.float32 or scaled.dtype == torch.float64
    # Every scaled value lies within [0.0, 1.0].
    assert torch.all(scaled >= 0.0)
    assert torch.all(scaled <= 1.0)
    # The mapping is exactly raw / 255.
    expected = raw.float() / 255.0
    assert torch.allclose(scaled, expected)


# ---------------------------------------------------------------------------
# Property 3 — Encoder produces a fixed-length feature vector
# ---------------------------------------------------------------------------
# A single encoder instance is reused across examples so "identical length across
# inputs" is meaningful (the feature_dim is a fixed property of the encoder).
_ENCODER_FEATURE_DIM = 256
_shared_encoder = CNNEncoder(feature_dim=_ENCODER_FEATURE_DIM).eval()


# Feature: ppo-car-racing, Property 3: Encoder produces a fixed-length feature vector
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_3_encoder_produces_fixed_length_feature_vector(
    batch_size: int, seed: int
) -> None:
    # Build a valid channels-first (N, 3, 96, 96) observation with random pixels,
    # scaled into [0.0, 1.0] exactly as the network expects at runtime.
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(
        0,
        256,
        (batch_size, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH),
        generator=generator,
        dtype=torch.uint8,
    )
    obs = scale_pixels(raw)

    with torch.no_grad():
        features = _shared_encoder(obs)

    # One feature vector per observation, each of length feature_dim.
    assert features.shape == (batch_size, _ENCODER_FEATURE_DIM)
    # The per-observation feature length is identical for every input.
    assert features.shape[-1] == _shared_encoder.feature_dim == _ENCODER_FEATURE_DIM


# ---------------------------------------------------------------------------
# Property 4 — Actor standard deviation is strictly positive
# ---------------------------------------------------------------------------
# Feature: ppo-car-racing, Property 4: Actor standard deviation is strictly positive
@settings(max_examples=120, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    action_dim=st.integers(min_value=1, max_value=6),
    feature_dim=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    # Drive log_std across a wide range (including large negatives, which produce
    # tiny std values) to stress the "strictly positive" guarantee.
    log_std_fill=st.floats(min_value=-30.0, max_value=10.0),
)
def test_property_4_actor_std_strictly_positive(
    batch_size: int,
    action_dim: int,
    feature_dim: int,
    seed: int,
    log_std_fill: float,
) -> None:
    torch.manual_seed(seed)
    actor = ActorHead(feature_dim=feature_dim, action_dim=action_dim).eval()
    # Overwrite the learnable log_std to exercise extreme values deterministically.
    with torch.no_grad():
        actor.log_std.fill_(log_std_fill)

    features = torch.randn(batch_size, feature_dim)

    with torch.no_grad():
        mean, std = actor(features)

    # mean and std vectors have length == action_dim per observation.
    assert mean.shape == (batch_size, action_dim)
    assert std.shape == (batch_size, action_dim)
    # Every std component is strictly greater than 0.
    assert torch.all(std > 0.0)


# ---------------------------------------------------------------------------
# Unit tests — encoder shape-mismatch rejection (Req 2.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_shape",
    [
        (1, 1, OBS_HEIGHT, OBS_WIDTH),       # wrong channel count
        (1, OBS_CHANNELS, 64, 64),           # wrong height/width
        (1, OBS_CHANNELS, OBS_HEIGHT, 95),   # wrong width
        (OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH),  # missing batch dim (3-D)
        (2, OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS),  # channels-last, not channels-first
    ],
)
def test_encoder_rejects_non_96x96x3_observation(bad_shape: tuple[int, ...]) -> None:
    encoder = CNNEncoder(feature_dim=256).eval()
    bad_obs = torch.zeros(bad_shape, dtype=torch.float32)

    # A shape mismatch must raise an error and produce no feature vector.
    with pytest.raises(ValueError) as exc_info:
        encoder(bad_obs)

    # The error names the shape mismatch so the failure is a teaching moment.
    assert "shape" in str(exc_info.value).lower()


def test_encoder_accepts_valid_96x96x3_observation() -> None:
    # Sanity companion to the rejection test: a correctly shaped observation is
    # accepted and yields a feature vector of the expected fixed length.
    encoder = CNNEncoder(feature_dim=256).eval()
    obs = torch.zeros(
        (2, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH), dtype=torch.float32
    )

    with torch.no_grad():
        features = encoder(obs)

    assert features.shape == (2, 256)


# ---------------------------------------------------------------------------
# Unit tests — Critic produces exactly one scalar per observation (Req 2.4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_critic_produces_one_scalar_per_observation(batch_size: int) -> None:
    feature_dim = 16
    critic = CriticHead(feature_dim=feature_dim).eval()
    features = torch.randn(batch_size, feature_dim)

    with torch.no_grad():
        values = critic(features)

    # Exactly one scalar per observation: shape (N,), one value each.
    assert values.shape == (batch_size,)
    assert values.numel() == batch_size
    # Each produced value is a finite scalar real number.
    assert torch.all(torch.isfinite(values))
    # Flattening does not change the count — confirms no extra trailing dims.
    assert values.reshape(-1).shape[0] == batch_size
