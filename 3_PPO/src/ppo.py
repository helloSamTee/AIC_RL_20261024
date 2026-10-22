"""The clipped PPO update — "how does a fresh rollout actually improve the policy?"

By the time we reach this module the hard conceptual work is done: we have a buffer
full of on-policy transitions (``rollout_buffer.py``), and we have advantages and
returns computed from them with GAE. This file is the step that *uses* that batch to
nudge the networks, and then the training loop will throw the batch away. It is the
"update" in the **collect → update → clear** rhythm.

Most of what happens here is ordinary PyTorch training glue: loop over epochs, slice
the batch into minibatches, run a forward pass, build a loss, call ``backward()``,
step the optimizer. To keep the *ideas* visible under that glue, the two pieces of
genuinely PPO-specific math are pulled out into their own small, separately testable
functions:

1. :func:`iter_minibatches` — the pure bookkeeping of chopping ``n`` transitions into
   shuffled minibatches so every transition is used exactly once per epoch.
2. :func:`clipped_surrogate` — PPO's signature policy loss, built from a probability
   ratio that is **clamped** so a single update can never move the policy too far.

The one idea worth pausing on is that **ratio clamp**. PPO compares the updated
policy against the policy that collected the data via a ratio
``r = exp(new_log_prob - old_log_prob)``: ``r > 1`` means the new policy makes the
taken action more likely, ``r < 1`` less likely. Left unchecked, a single batch could
push ``r`` arbitrarily far and lurch the policy somewhere it will regret. The clipped
surrogate caps how much a step can gain from moving ``r`` outside
``[1 - clip_range, 1 + clip_range]``, so each update takes a modest, *proximal* step —
that cautiousness is literally the "Proximal" in Proximal Policy Optimization.

The loss that drives the update combines three terms:

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

* ``policy_loss`` — the clipped surrogate (we *minimize* its negative-min form, which
  *maximizes* the clipped objective).
* ``value_loss`` — how far the critic's value estimate is from the GAE return; this
  trains the critic to predict returns better.
* ``entropy`` — a measure of how random the policy still is. We *subtract* it (scaled
  by ``entropy_coef``) so that minimizing the total loss gently *rewards* staying
  exploratory, which keeps the policy from collapsing onto one action too early.

All three networks — the shared CNN encoder, the Actor, and the Critic — are updated
together from that single combined loss, because the encoder's features feed both
heads and should learn from both signals at once.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from networks import ActorHead, CNNEncoder, CriticHead, scale_pixels
from policy import tanh_log_prob


def iter_minibatches(n: int, minibatch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle the indices ``0..n-1`` and split them into minibatches.

    Within one update epoch PPO learns from the whole rollout, but in small
    randomly-ordered chunks rather than all at once. This function does exactly
    that bookkeeping and nothing else, which keeps it easy to reason about and to
    test:

    1. Produce a random permutation of the indices ``0, 1, ..., n-1`` using the
       supplied NumPy ``rng`` (passing the generator in keeps runs reproducible —
       the same seed reshuffles the same way).
    2. Slice that shuffled list into consecutive chunks of ``minibatch_size``. The
       final chunk simply holds whatever is left over when ``n`` is not an exact
       multiple of ``minibatch_size``.

    The result is an **exact cover** of the indices: every index ``0..n-1`` appears
    in exactly one minibatch, no index is missing, and none is repeated (Req 6.2).
    The caller does one gradient step per returned minibatch.

    Returns a list of 1-D NumPy arrays of indices, in the shuffled order.
    """

    # Step 1: a reproducible random ordering of all n indices. rng.permutation(n)
    # returns a shuffled array containing each of 0..n-1 exactly once.
    shuffled_indices = rng.permutation(n)

    # Step 2: walk the shuffled indices in strides of minibatch_size, carving out
    # one minibatch per stride. The last slice naturally holds the remainder when
    # n is not an exact multiple of minibatch_size (Python slicing just stops at
    # the end of the array), so every index lands in exactly one minibatch.
    minibatches: list[np.ndarray] = []
    for start in range(0, n, minibatch_size):
        minibatch = shuffled_indices[start : start + minibatch_size]
        minibatches.append(minibatch)

    return minibatches


def clipped_surrogate(ratio: Tensor, advantage: Tensor, clip_range: float) -> Tensor:
    """PPO's clipped policy loss, computed per transition (the caller averages).

    This is the heart of PPO. ``ratio`` is the per-transition probability ratio
    between the updated policy and the policy that collected the data,
    ``ratio = exp(new_log_prob - old_log_prob)``: a value above 1 means the new
    policy makes the taken action *more* likely, below 1 means *less* likely.
    ``advantage`` says whether that action turned out better (positive) or worse
    (negative) than the critic expected.

    A plain policy-gradient objective would be ``ratio * advantage`` — push the
    probability of good actions up, bad actions down. The danger is that a single
    batch could drive ``ratio`` arbitrarily far and move the policy too much in one
    step. PPO's fix is to also build a **clamped** version of the ratio, pinned to
    ``[1 - clip_range, 1 + clip_range]``, and take the *pessimistic* (smaller) of
    the two objectives:

        objective = min( ratio * advantage,  clamp(ratio) * advantage )

    Taking the minimum removes any incentive to push the ratio outside the clip
    band: once past the edge, the clamped term stops improving, so the step gains
    nothing by moving the policy further. That is what limits how far each update
    moves the policy — the "proximal" guarantee (Req 6.3).

    Optimizers *minimize*, but we want to *maximize* that objective, so this
    returns its **negative**: ``-min(ratio*A, clamp(ratio)*A)``. The value is
    returned per transition; the caller takes the mean over the minibatch.
    """

    # The unclipped term: the raw policy-gradient surrogate.
    unclipped_objective = ratio * advantage

    # The clipped term: pin the ratio into [1 - clip_range, 1 + clip_range] first,
    # then weight by the advantage. Beyond the band the clamped ratio is constant,
    # so this term stops rewarding further movement of the policy.
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped_objective = clipped_ratio * advantage

    # Take the pessimistic (elementwise minimum) of the two, then negate so that
    # minimizing this loss maximizes the clipped PPO objective.
    surrogate_objective = torch.min(unclipped_objective, clipped_objective)
    return -surrogate_objective


class PPOUpdater:
    """Runs the clipped PPO update over one collected rollout.

    Construct it once with the three networks (the shared CNN encoder, the Actor,
    and the Critic), the :class:`~config.Config`, and the compute device. It builds
    a single Adam optimizer over the parameters of *all three* networks, because the
    encoder's features feed both heads and should learn from the combined loss at
    once.

    Call :meth:`update` once per training iteration, right after GAE has produced the
    advantages and returns for the freshly collected buffer. It runs
    ``config.update_epochs`` passes over the batch, doing one gradient step per
    minibatch, and returns the mean policy loss, value loss, and entropy for logging.
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        actor: ActorHead,
        critic: CriticHead,
        config,
        device: torch.device,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = device

        # One optimizer over all three networks' parameters. Chaining the parameter
        # lists means a single optimizer.step() updates the encoder, Actor, and
        # Critic together from the one combined loss.
        all_parameters = (
            list(encoder.parameters())
            + list(actor.parameters())
            + list(critic.parameters())
        )
        self.optimizer = torch.optim.Adam(all_parameters, lr=config.learning_rate)

        # A NumPy generator drives the minibatch shuffling. Accepting one (or making
        # a default-seeded one) keeps the partitioning reproducible across runs.
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)

    def update(
        self,
        buffer,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        """Run ``update_epochs`` passes over the rollout and return mean losses.

        ``buffer`` is the filled :class:`~rollout_buffer.RolloutBuffer`;
        ``advantages`` and ``returns`` are the (already normalized) GAE outputs, one
        per stored transition, as NumPy arrays.

        Each epoch partitions *all* stored transitions into minibatches via
        :func:`iter_minibatches` and does exactly one gradient step per minibatch
        (Req 6.1, 6.2). For each minibatch we recompute the current policy's
        log-prob of the stored action, form the probability ratio against the stored
        old log-prob, and build the combined loss
        ``policy_loss + value_coef * value_loss - entropy_coef * entropy`` (Req 6.4,
        6.5). Gradients are clipped to ``max_grad_norm`` before the optimizer step
        when that clip is configured, and left unclipped otherwise (Req 6.6, 6.7).

        Returns a dict ``{"policy_loss", "value_loss", "entropy"}`` of the means
        across every minibatch step, for TensorBoard logging (Req 8.2).
        """

        config = self.config
        number_of_steps = len(buffer)

        # --- Pull the stored batch onto the device, once ----------------------
        # Observations are stored channels-LAST (N, 96, 96, 3) as raw uint8. The
        # encoder wants scaled, channels-FIRST (N, 3, 96, 96) input, so we scale
        # the pixels to 0..1 and then permute the axes.
        stored_obs_uint8 = torch.as_tensor(
            buffer._observations[:number_of_steps], device=self.device
        )
        scaled_obs = scale_pixels(stored_obs_uint8)
        observations = scaled_obs.permute(0, 3, 1, 2).contiguous()

        # The raw (pre-squash) actions we actually took, and the log-probs they had
        # under the collecting policy. We recompute new log-probs from these raw
        # actions to form the PPO ratio.
        stored_actions = torch.as_tensor(
            buffer._actions[:number_of_steps], device=self.device
        )
        old_log_probs = torch.as_tensor(
            buffer._log_probs[:number_of_steps], device=self.device
        )

        # GAE outputs, moved onto the device as tensors.
        advantages_tensor = torch.as_tensor(
            np.asarray(advantages, dtype=np.float32), device=self.device
        )
        returns_tensor = torch.as_tensor(
            np.asarray(returns, dtype=np.float32), device=self.device
        )

        # Running sums so we can report the mean of each loss term across every
        # minibatch gradient step at the end.
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        number_of_minibatch_steps = 0

        # --- Epoch loop: reuse the same on-policy batch update_epochs times ----
        for _ in range(config.update_epochs):
            # Fresh shuffle each epoch; every transition lands in exactly one
            # minibatch (an exact cover of 0..n-1).
            minibatches = iter_minibatches(number_of_steps, config.minibatch_size, self.rng)

            for minibatch_indices in minibatches:
                # Index this minibatch out of the batch tensors. A LongTensor index
                # keeps everything on-device.
                index = torch.as_tensor(
                    minibatch_indices, dtype=torch.long, device=self.device
                )
                obs_minibatch = observations[index]
                actions_minibatch = stored_actions[index]
                old_log_prob_minibatch = old_log_probs[index]
                advantage_minibatch = advantages_tensor[index]
                return_minibatch = returns_tensor[index]

                # --- Forward pass: encoder -> both heads ----------------------
                features = self.encoder(obs_minibatch)
                mean, std = self.actor(features)
                value_estimate = self.critic(features)

                # --- Policy loss: the clipped surrogate -----------------------
                # Recompute the current policy's log-prob of the action we took,
                # then form the probability ratio against the stored old log-prob.
                new_log_prob = tanh_log_prob(mean, std, actions_minibatch)
                ratio = torch.exp(new_log_prob - old_log_prob_minibatch)
                policy_loss = clipped_surrogate(
                    ratio, advantage_minibatch, config.clip_range
                ).mean()

                # --- Value loss: critic estimate vs. GAE return ---------------
                # Mean squared error trains the critic to predict the returns.
                value_loss = nn.functional.mse_loss(value_estimate, return_minibatch)

                # --- Entropy bonus --------------------------------------------
                # Entropy of a diagonal Gaussian, per action dimension, is
                #   0.5 * log(2*pi*e*std^2) = 0.5 + 0.5*log(2*pi) + log(std).
                # Sum over the action dimensions (independent dims add) and average
                # over the minibatch. Higher entropy = a more exploratory policy.
                entropy_per_dim = 0.5 + 0.5 * np.log(2.0 * np.pi) + torch.log(std)
                entropy = entropy_per_dim.sum(dim=-1).mean()

                # --- Combine into one total loss and step ---------------------
                # Subtracting the entropy term means minimizing the total loss
                # rewards staying exploratory.
                total_loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()

                # Clip the combined gradient to max_grad_norm before stepping when
                # a positive clip is configured; otherwise step unclipped.
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    clip_parameters = (
                        list(self.encoder.parameters())
                        + list(self.actor.parameters())
                        + list(self.critic.parameters())
                    )
                    nn.utils.clip_grad_norm_(clip_parameters, config.max_grad_norm)

                self.optimizer.step()

                # Accumulate the scalar value of each term for mean logging.
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                number_of_minibatch_steps += 1

        # Report the mean of each loss term across all minibatch gradient steps.
        return {
            "policy_loss": total_policy_loss / number_of_minibatch_steps,
            "value_loss": total_value_loss / number_of_minibatch_steps,
            "entropy": total_entropy / number_of_minibatch_steps,
        }
