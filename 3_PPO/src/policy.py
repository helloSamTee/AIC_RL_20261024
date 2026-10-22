"""The continuous-action policy — "how does the agent pick a real-valued action?"

DQN picked an action by taking the ``argmax`` over a handful of discrete Q-values.
CarRacing is different: it wants a **continuous** action — three real numbers
(steering, gas, brake), each inside its own ``[low, high]`` range. There is no
finite list of actions to argmax over, so PPO describes its policy as a
**Gaussian** (a bell curve) over each action dimension and *samples* from it.

That sampling pipeline is the subject of this module, and it happens in three
moves:

1. **Sample** a raw, unbounded action from the Gaussian ``N(mean, std)`` the Actor
   produced. Call that pre-squash sample ``u``. It can be any real number.
2. **Squash** ``u`` through ``tanh`` so it lands in ``[-1, 1]`` no matter how large
   it was, then **rescale** that into the environment's real action bounds
   ``[low, high]``. This is how we guarantee a sampled action is always legal.
3. **Clamp** the final action to the bounds, so floating-point rounding can never
   nudge a component a hair past its limit.

The subtle part is the **log-probability**. PPO's update needs to know how likely
the action it took was under the policy. But we did not just sample from a plain
Gaussian — we squashed the sample through ``tanh``. Whenever you push a random
variable through a nonlinear function you have to correct its probability density
with a **change-of-variables** term; otherwise the log-prob is simply wrong. For a
``tanh`` squash that correction is ``-sum(log(1 - tanh(u)^2))`` over the action
dimensions.

That correction has a nasty numerical trap. As ``tanh(u)`` saturates toward
``±1`` (which happens for perfectly ordinary large-ish ``u``), ``1 - tanh(u)^2``
races toward 0, and ``log(0)`` is ``-inf``. A single ``-inf`` or ``NaN`` poisons
the whole PPO update. So instead of computing ``log(1 - tanh(u)^2)`` directly we
use the mathematically identical but numerically **stable** identity

    log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))

which stays finite even deep into saturation (Req 3.3). Deriving it is a short
exercise; the payoff is a log-prob that never blows up.

Finally, evaluation is special. During training we *sample* (we want exploration —
some randomness to discover good driving). During evaluation we want to see the
policy's best guess with **exploration off**, so :func:`deterministic_action` skips
the sampling entirely and just squashes the ``mean`` — no noise, fully repeatable
(Req 3.4, 11.1).

Every piece of math below lives in its own small, separately testable function,
because this is the numerically delicate corner of the whole project.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def tanh_rescale(raw_action: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """Squash ``raw_action`` through tanh into ``[-1, 1]``, then rescale onto ``[low, high]``.

    This is the "make any real number into a legal action" step. The raw,
    pre-squash action ``u`` can be any real number (it came out of a Gaussian).
    Two moves turn it into something the environment will accept:

    1. ``tanh(u)`` maps the whole real line into the open interval ``(-1, 1)``.
       Huge positive ``u`` approaches ``+1``; huge negative ``u`` approaches ``-1``.
    2. Linearly rescale that ``[-1, 1]`` value onto the per-dimension bounds
       ``[low, high]``. The formula is the standard affine map from ``[-1, 1]``:

           action = low + (squashed + 1) * 0.5 * (high - low)

       When ``squashed == -1`` this gives ``low``; when ``squashed == +1`` this
       gives ``high``; and ``0`` lands exactly at the midpoint.

    ``low`` and ``high`` are per-dimension bounds (for CarRacing, three values
    each), so the rescale is applied component-wise. The returned action lies in
    ``[low, high]`` up to floating-point rounding — :func:`clamp_to_bounds` mops
    up that last little bit of rounding error (Req 3.2).
    """

    # Step 1: squash onto (-1, 1). No matter how extreme raw_action is, tanh keeps
    # the result inside [-1, 1].
    squashed = torch.tanh(raw_action)

    # Step 2: affine rescale from [-1, 1] onto [low, high], component-wise.
    # (squashed + 1) * 0.5 remaps [-1, 1] -> [0, 1]; multiplying by the width
    # (high - low) and adding low then stretches that onto [low, high].
    span = high - low
    rescaled_action = low + (squashed + 1.0) * 0.5 * span
    return rescaled_action


def tanh_log_prob(mean: Tensor, std: Tensor, raw_action: Tensor) -> Tensor:
    """Log-probability of ``raw_action`` under ``N(mean, std)`` plus the tanh correction.

    PPO needs the log-probability of the action the policy took. Because the
    action was produced by sampling ``u`` from the Gaussian ``N(mean, std)`` and
    then squashing it through ``tanh``, this log-prob has **two parts**:

    1. The ordinary Gaussian log-density of the pre-squash sample ``u`` under
       ``N(mean, std)``, summed over the action dimensions.
    2. The ``tanh`` **change-of-variables correction**. Pushing a random variable
       through ``tanh`` compresses probability mass near ``±1``, and the density
       must be corrected by subtracting ``log(1 - tanh(u)^2)`` for each dimension.

    The numerical catch (and why we do not write ``log(1 - tanh(u)**2)`` directly):
    as ``tanh(u)`` saturates toward ``±1``, ``1 - tanh(u)^2`` underflows to 0 and
    its log becomes ``-inf``, which would poison the whole update. We instead use
    the algebraically identical but stable identity

        log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))

    which stays finite for any finite ``u`` (``softplus`` never overflows here).
    The correction is summed over the action dimensions so the returned log-prob
    is one scalar per sampled action, and it is always finite (Req 3.3).
    """

    # --- Part 1: Gaussian log-density of the pre-squash sample ------------
    # The log-density of u under N(mean, std) for one dimension is:
    #   -log(std) - 0.5 * log(2*pi) - 0.5 * ((u - mean) / std)^2
    # We compute it component-wise, then sum across the action dimensions so each
    # sampled action contributes a single scalar.
    variance = std * std
    squared_error = (raw_action - mean) ** 2
    gaussian_log_density = (
        -torch.log(std)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * squared_error / variance
    )
    gaussian_log_prob = gaussian_log_density.sum(dim=-1)

    # --- Part 2: the tanh change-of-variables correction ------------------
    # Mathematically this is sum(log(1 - tanh(u)^2)), but computed with the stable
    # identity below so it never overflows to -inf near saturation.
    #   log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
    # softplus(x) = log(1 + exp(x)) is provided by torch in a numerically safe form.
    log_two = math.log(2.0)
    correction_per_dim = 2.0 * (log_two - raw_action - F.softplus(-2.0 * raw_action))
    tanh_correction = correction_per_dim.sum(dim=-1)

    # The squashed action's log-prob is the Gaussian log-prob MINUS the correction.
    # (Subtracting the correction is the change-of-variables adjustment.)
    log_prob = gaussian_log_prob - tanh_correction
    return log_prob


def sample_action(mean: Tensor, std: Tensor, rng: torch.Generator) -> tuple[Tensor, Tensor]:
    """Sample a pre-squash action from ``N(mean, std)`` and return ``(raw_action, log_prob)``.

    This is the action-selection step used during **data collection** (training),
    where we *want* randomness so the agent explores. It draws a pre-squash sample
    ``u`` from the Gaussian the Actor produced and reports both:

    - ``raw_action`` — the pre-squash sample ``u`` itself. We return the raw sample
      (not the squashed/rescaled action) because the log-prob and later PPO
      recomputation are both defined in terms of ``u``. The caller turns ``u`` into
      an actual environment action with :func:`tanh_rescale` and
      :func:`clamp_to_bounds`.
    - ``log_prob`` — the log-probability of that action, already including the
      ``tanh`` change-of-variables correction (via :func:`tanh_log_prob`), so it is
      finite and directly usable by the PPO update (Req 3.1, 3.3).

    Sampling is done with the standard reparameterization ``u = mean + std * eps``,
    where ``eps`` is standard-normal noise drawn from the supplied ``rng``. Passing
    an explicit ``torch.Generator`` keeps runs reproducible: the same seed yields
    the same action sequence (Req 13.2).
    """

    # Draw standard-normal noise shaped like the mean, using the provided generator
    # so the randomness is seeded and reproducible.
    standard_normal_noise = torch.randn(
        mean.shape, generator=rng, dtype=mean.dtype, device=mean.device
    )

    # Reparameterized sample: shift the noise by the mean and scale by the std.
    # This draws u ~ N(mean, std) one component at a time.
    raw_action = mean + std * standard_normal_noise

    # Compute the (tanh-corrected, always finite) log-probability of this sample.
    log_prob = tanh_log_prob(mean, std, raw_action)

    return raw_action, log_prob


def deterministic_action(mean: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """The evaluation action — **exploration off**: ``tanh(mean)`` rescaled onto ``[low, high]``.

    During evaluation we want to measure the policy's *best guess*, not its
    exploratory wandering. So this path adds **no sampled noise at all**: it takes
    the Actor's ``mean`` directly, squashes it through ``tanh``, and rescales onto
    the action bounds — exactly the same squash-and-rescale as the sampling path,
    but with the random draw removed.

    Because there is no randomness, this is fully **deterministic**: the same
    ``mean`` and bounds always yield the same action, which is what makes an
    evaluation run repeatable (Req 3.4, 11.1). This is the single "exploration off"
    entry point the evaluation pass reuses.
    """

    # No sampling: feed the mean straight through the same squash-and-rescale the
    # sampling path uses. tanh(mean) is the policy's central (most-likely) action.
    return tanh_rescale(mean, low, high)


def clamp_to_bounds(action: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """Clamp each action component into the inclusive bounds ``[low, high]``.

    The ``tanh`` rescale already lands actions inside ``[low, high]`` in exact
    arithmetic, but floating-point rounding can leave a component a vanishingly
    small amount past a bound. The environment expects strictly legal actions, so
    this final guard clamps every component component-wise: anything below ``low``
    becomes ``low``, anything above ``high`` becomes ``high`` (Req 3.5).

    ``torch.clamp`` with tensor bounds applies the clamp per element, honouring the
    per-dimension ``low``/``high`` for each action component.
    """

    # Component-wise clamp: pin each element into its own [low, high] band. This
    # is the last line of defence against floating-point overshoot.
    clamped_action = torch.clamp(action, min=low, max=high)
    return clamped_action
