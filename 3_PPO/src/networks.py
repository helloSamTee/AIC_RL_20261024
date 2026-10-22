"""The CNN encoder and the two heads — "how does the agent see, act, and judge?"

CarRacing hands the agent a raw **96x96x3 RGB image** at every step and expects a
**continuous action** back (for CarRacing that is three numbers: steering, gas,
brake). This module holds the three neural-network pieces that turn pixels into
an action distribution and a value estimate:

1. ``scale_pixels`` — a tiny standalone function that rescales raw integer pixels
   (0..255) into floats (0.0..1.0) before anything touches the network. Neural
   networks train far more happily on small, normalized inputs than on raw bytes.

2. ``CNNEncoder`` — a small convolutional network that compresses one 96x96x3
   frame into a short, fixed-length **feature vector**. This is the agent's "eyes".

3. ``ActorHead`` and ``CriticHead`` — the two heads that read that feature vector.
   The Actor turns features into a Gaussian action distribution (a ``mean`` and a
   ``std`` per action dimension); the Critic turns the same features into a single
   number estimating "how good is this state?".

The one idea to hold onto: **the encoder is SHARED.** We run the encoder once per
observation to get a single feature vector, and then *both* heads read that same
vector. The Actor and Critic do not each have their own private view of the image;
they share one set of eyes. That saves computation and lets both heads benefit
from the same learned visual features.

A note on the two problem-shape choices, since this is where PPO differs from the
companion DQN module:

- **Pixels are scaled before the CNN.** DQN fed a short state vector straight into
  an MLP. Here every frame is divided by 255 first (``scale_pixels``) so values
  land in 0.0..1.0 before entering the convolutions.
- **The std comes from exp(log_std).** A standard deviation must be strictly
  positive — a Gaussian with std <= 0 is nonsense. Rather than try to force a
  network output to stay positive, the Actor keeps a *learnable* ``log_std`` that
  can be any real number, and exponentiates it: ``std = exp(log_std)``. Because
  ``exp`` of any finite number is strictly greater than 0, every std we produce is
  guaranteed positive, for free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# The observation geometry CarRacing produces. Kept as named constants so the
# shape check below reads clearly and a learner can see exactly what is expected.
OBS_CHANNELS = 3
OBS_HEIGHT = 96
OBS_WIDTH = 96


def scale_pixels(obs_uint8: Tensor) -> Tensor:
    """Scale raw pixels from the integer range 0..255 into the float range 0.0..1.0.

    A raw RGB frame arrives as whole numbers from 0 (black) to 255 (full
    intensity). Neural networks learn more stably from small, normalized inputs,
    so before an observation ever enters the CNN we divide every pixel by 255.
    A pixel of 0 maps to 0.0, a pixel of 255 maps to 1.0, and everything in
    between lands proportionally in [0.0, 1.0].

    Keeping this as a tiny standalone function (rather than burying it inside the
    encoder) makes the scaling rule obvious and easy to check on its own.
    """

    # .float() converts the integer tensor to floating point; dividing by 255.0
    # then squeezes the 0..255 range down into 0.0..1.0.
    return obs_uint8.float() / 255.0


class CNNEncoder(nn.Module):
    """Map a 96x96x3 observation to a fixed-length feature vector (the shared "eyes").

    This is the convolutional network both heads rely on. It expects a batch of
    observations in **channels-first** layout ``(N, 3, 96, 96)`` — that is, N
    images, each with 3 colour channels of height 96 and width 96 — and already
    scaled to 0.0..1.0 by :func:`scale_pixels`. It returns one feature vector of
    length ``feature_dim`` per image, i.e. shape ``(N, feature_dim)``.

    The length of that feature vector is the SAME for every valid observation —
    that is what "fixed-length" means and why both heads can always consume it.

    If an observation does not have the expected channel/height/width, the
    ``forward`` pass raises a plain ``ValueError`` naming the expected versus the
    actual shape, and does **not** produce a feature vector. Catching a bad shape
    here, with a clear message, beats a confusing failure deep inside a conv layer.
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()

        # Remember the output width so the heads (and tests) can read it.
        self.feature_dim = feature_dim

        # A small stack of convolutions that progressively shrink the 96x96 image
        # while growing the number of channels — the standard "see broader, coarser
        # patterns at each step" pattern. Each conv is followed by a ReLU, which
        # keeps positive signals and zeros out negatives (the usual nonlinearity).
        #
        # Shapes as the image flows through (for a single image, ignoring batch):
        #   input         3 x 96 x 96
        #   conv1 (s=4)  32 x 23 x 23
        #   conv2 (s=2)  64 x 10 x 10
        #   conv3 (s=1)  64 x  8 x  8
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=OBS_CHANNELS, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # After the convolutions each image is a 64 x 8 x 8 block of numbers. We
        # flatten that block into one long vector and run it through a linear layer
        # to produce the fixed-length feature vector the heads consume.
        conv_output_size = 64 * 8 * 8
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, feature_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """Run a batch of (already scaled) observations through the encoder.

        ``obs`` must have shape ``(N, 3, 96, 96)``. The return value has shape
        ``(N, feature_dim)`` — one fixed-length feature vector per observation.
        """

        # --- Validate the shape BEFORE doing any work (Req 2.2) ---------------
        # We only accept a channels-first batch of 96x96x3 images. If the channel,
        # height, or width dimensions are wrong, stop immediately with a clear
        # message naming what we expected versus what we got, and produce nothing.
        expected_shape = (OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH)
        actual_per_image_shape = tuple(obs.shape[1:])
        if obs.dim() != 4 or actual_per_image_shape != expected_shape:
            raise ValueError(
                "CNNEncoder expected a channels-first batch of observations with "
                f"per-image shape {expected_shape} (i.e. a 4-D tensor (N, 3, 96, 96)), "
                f"but got a tensor of shape {tuple(obs.shape)}."
            )

        # --- Forward pass, written as an explicit sequence of steps -----------
        # Raw image -> convolutional feature maps -> flattened -> feature vector.
        conv_features = self.conv_layers(obs)
        features = self.fully_connected(conv_features)
        return features


class ActorHead(nn.Module):
    """Features -> a Gaussian action distribution: a ``mean`` and a ``std``.

    The Actor reads the shared feature vector and describes *how to act* as a
    Gaussian (bell-curve) distribution over each action dimension. It outputs two
    vectors, each of length ``action_dim``:

    - ``mean`` — the centre of the distribution for each action component, read
      straight from a linear layer (it may be any real number).
    - ``std`` — the spread (standard deviation) of the distribution for each
      component, which MUST be strictly positive.

    Guaranteeing a positive std: a standard deviation of zero or below is not a
    valid Gaussian. Instead of trying to clamp a raw network output, the Actor
    keeps a single **learnable ``log_std``** parameter — one unbounded real number
    per action dimension — and exponentiates it: ``std = exp(log_std)``. Since
    ``exp`` of any finite value is always greater than 0, every std is strictly
    positive no matter what the network learns (Req 2.3). The ``log_std`` is a
    *state-independent* learnable parameter, which is the common PPO choice: the
    policy's "mean" reacts to what it sees, while its overall exploration spread is
    learned globally.
    """

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()

        self.action_dim = action_dim

        # The mean head: features -> one mean per action dimension. No activation,
        # because the mean is allowed to be any real number.
        self.mean_layer = nn.Linear(feature_dim, action_dim)

        # The log-std: a single learnable vector of length action_dim, independent
        # of the current observation. Initialized to 0 so the starting std is
        # exp(0) = 1 (a reasonable, moderate amount of initial exploration).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Turn a batch of feature vectors into ``(mean, std)``.

        ``features`` has shape ``(N, feature_dim)``. Both returned tensors have
        shape ``(N, action_dim)``, and every value in ``std`` is strictly > 0.
        """

        # Mean: one value per action dimension, straight from the linear layer.
        mean = self.mean_layer(features)

        # Std: exponentiate the learnable log_std so it is strictly positive, then
        # expand it to match the batch size so each observation gets the same std
        # vector (the spread is state-independent, as described above).
        std = torch.exp(self.log_std)
        std = std.expand_as(mean)

        return mean, std


class CriticHead(nn.Module):
    """Features -> a single scalar value estimate per observation (Req 2.4).

    The Critic reads the same shared feature vector and answers one question:
    "starting from this state, how much total reward do I expect to collect?"
    That answer is a single number (a scalar) per observation. During training it
    becomes the baseline the Actor's advantages are measured against.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()

        # A single linear output: features -> one number. No activation, because a
        # value estimate can be any real number (positive or negative).
        self.value_layer = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Turn a batch of feature vectors into one scalar value each.

        ``features`` has shape ``(N, feature_dim)``. The return value has shape
        ``(N,)`` — exactly one scalar state-value estimate per observation.
        """

        # The linear layer gives shape (N, 1); squeeze the trailing dimension so
        # we return a clean (N,) vector of one scalar per observation.
        value = self.value_layer(features)
        return value.squeeze(-1)
