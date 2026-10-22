"""The rollout buffer and GAE — "why is there no replay buffer here?"

This is the one place where PPO's **on-policy** nature is most visible, so read
the distinction carefully. The companion DQN project kept a large, long-lived
``ReplayBuffer``: a ring that held tens of thousands of past transitions and was
sampled at random, reusing old experience over and over, overwriting only the
oldest entry when it filled. That works because DQN is *off-policy* — its update
math is valid for experience collected by any earlier policy.

PPO cannot do that. Its update is only correct for data collected by the
**current** policy. So the buffer here behaves in the opposite way:

* It **never replays.** We do not sample random slices of history out of it.
* It **never evicts.** There is no ring, no wrap-around, no overwriting the oldest
  entry. It fills forward to exactly ``rollout_horizon`` and stops.
* It is **single-use.** Once full it is consumed *once* by the update phase, and
  then :meth:`clear` empties it completely before the next collection phase.

The rhythm of the whole training loop is therefore: **collect → update → clear**,
repeat. This buffer is the short-lived scratchpad for one lap of that rhythm.

What it stores per step (one ``Transition``):

* ``obs``      — the raw ``uint8`` 96x96x3 image the action was chosen on. Stored
  as ``uint8`` (not float) to save memory; it is scaled to 0.0..1.0 only when fed
  to the network, never in here.
* ``action``   — the ``float32`` action vector taken.
* ``log_prob`` — the ``float32`` log-probability of that action under the policy
  that collected it (PPO compares the updated policy against this later).
* ``reward``   — the ``float32`` reward received.
* ``terminal`` — a ``bool`` that is True only on a true terminal step (episode
  genuinely ended), and False on a truncation (episode cut short by a time limit).
* ``value``    — the ``float32`` critic value estimate for ``obs``.

The second job of this module is **Generalized Advantage Estimation (GAE)**, which
turns the stored rewards and value estimates into two learning targets: an
*advantage* per step (how much better the action was than the critic expected) and
a *return* per step (the critic's regression target). GAE is computed in a single
**reverse** pass over the stored transitions — see :meth:`RolloutBuffer.compute_gae`
for the step-by-step recurrence and the role of the ``(1 - terminal)`` mask.

Finally, :func:`normalize_advantages` standardizes the advantages to zero mean and
unit variance before they enter the PPO objective, which keeps the policy-gradient
step well-scaled. It guards carefully against a zero-variance batch and leaves a
single-element batch untouched.
"""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    """A fixed-horizon, single-use store of on-policy transitions.

    The buffer is backed by preallocated NumPy arrays, one per transition field,
    all sized to exactly ``rollout_horizon``. Transitions are appended in order
    with :meth:`add` until the horizon is reached; the buffer never wraps around
    and never overwrites, which is the on-policy difference from the DQN replay
    buffer. After the update phase consumes the buffer, :meth:`clear` resets it to
    hold no transitions so the next collection phase starts fresh.

    Observations are stored as ``uint8`` in their raw ``(96, 96, 3)`` form to keep
    memory down; the network scales them to 0.0..1.0 at use time, not here.
    """

    def __init__(
        self,
        rollout_horizon: int,
        action_dim: int,
        observation_shape: tuple[int, ...] = (96, 96, 3),
    ) -> None:
        # ``rollout_horizon`` is the fixed number of transitions this buffer holds
        # for one collect -> update -> clear cycle. The arrays below are sized to
        # it exactly and are never grown or wrapped.
        self.rollout_horizon = rollout_horizon
        self.action_dim = action_dim
        self.observation_shape = observation_shape

        # ``_size`` is how many transitions are currently stored (0..horizon). It
        # only ever counts up during a collection phase; clear() resets it to 0.
        # There is deliberately no write-index-with-wraparound here — unlike the
        # DQN ring buffer, this buffer fills straight through and stops.
        self._size = 0

        # Preallocate one array per transition field, sized to the horizon.
        #   observations: raw uint8 images, shape (horizon, 96, 96, 3)
        #   actions:      float32 vectors,  shape (horizon, action_dim)
        #   log_probs:    float32 scalars,  shape (horizon,)
        #   rewards:      float32 scalars,  shape (horizon,)
        #   terminals:    bool flags,       shape (horizon,)
        #   values:       float32 scalars,  shape (horizon,)
        self._observations = np.zeros(
            (rollout_horizon, *observation_shape), dtype=np.uint8
        )
        self._actions = np.zeros((rollout_horizon, action_dim), dtype=np.float32)
        self._log_probs = np.zeros(rollout_horizon, dtype=np.float32)
        self._rewards = np.zeros(rollout_horizon, dtype=np.float32)
        self._terminals = np.zeros(rollout_horizon, dtype=np.bool_)
        self._values = np.zeros(rollout_horizon, dtype=np.float32)

    def __len__(self) -> int:
        """Return how many transitions are currently stored (0 up to the horizon)."""
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        terminal: bool,
        value: float,
    ) -> None:
        """Append one transition at the next free slot (no wrap-around, no eviction).

        The transition ``(obs, action, log_prob, reward, terminal, value)`` is
        written at index ``_size``, and the stored count then grows by one. Because
        this buffer fills straight through to ``rollout_horizon`` and is cleared
        between collection phases, writing at ``_size`` always lands in the next
        empty slot — there is no oldest entry to overwrite.

        Appending beyond the horizon is a programming error (collection should stop
        exactly at the horizon), so we raise rather than silently discard or wrap.
        """

        # Guard: the buffer is fixed-horizon and single-use. If collection tried to
        # push past the horizon, that is a bug in the training loop, not something
        # to paper over by wrapping around (which would silently corrupt the
        # on-policy batch). Fail loudly with a clear message.
        if self._size >= self.rollout_horizon:
            raise ValueError(
                "RolloutBuffer is full: it holds exactly rollout_horizon "
                f"({self.rollout_horizon}) transitions and does not evict or wrap. "
                "Collection should stop at the horizon; call clear() before reusing."
            )

        # Write the transition's fields into the next free slot. NumPy copies the
        # values into the preallocated arrays, so the buffer keeps its own copy and
        # does not hold references to the caller's objects.
        index = self._size
        self._observations[index] = obs
        self._actions[index] = action
        self._log_probs[index] = log_prob
        self._rewards[index] = reward
        self._terminals[index] = terminal
        self._values[index] = value

        # Advance the stored count by one. No modulo, no wrap: the index simply
        # marches forward to the horizon.
        self._size += 1

    def clear(self) -> None:
        """Empty the buffer so it holds no transitions before the next phase.

        Clearing is what makes the buffer *single-use*: after the update phase has
        consumed this batch, we reset the stored count to 0 so the next collection
        phase starts from an empty buffer. We only reset the count — the backing
        arrays stay allocated and will simply be overwritten slot-by-slot by the
        next round of :meth:`add` calls, so stale values can never be read back
        because :meth:`__len__` now reports 0 transitions.
        """

        # Reset the count to 0. The preallocated arrays are reused as-is; any old
        # contents are unreachable because nothing reads past ``_size``.
        self._size = 0

    def compute_gae(
        self,
        gamma: float,
        gae_lambda: float,
        bootstrap_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns in a single reverse-temporal pass.

        Generalized Advantage Estimation turns the stored rewards and critic value
        estimates into two per-step learning targets:

        * the **advantage** ``A_t`` — how much better the action taken at step ``t``
          turned out than the critic expected, and
        * the **return** ``R_t`` — the critic's regression target, computed as
          ``A_t + V(s_t)`` (advantage plus the stored value, Req 5.4).

        The computation walks the stored transitions from the **last** step back to
        the first (Req 5.1), because each step's advantage depends on the one after
        it. Two quantities are named explicitly as we go:

        * ``next_value`` — the critic's value of the state that *follows* step ``t``,
          i.e. ``V(s_{t+1})``. For the final stored step there is no next stored
          transition, so we bootstrap with ``bootstrap_value`` — the critic's
          estimate of the observation that comes right after the last stored step
          (Req 5.3).
        * ``advantage`` — the running advantage ``A_{t+1}`` carried backward from the
          later step; it starts at 0 beyond the final step.

        The per-step recurrence (Req 5.1, 5.2):

            delta_t = reward_t + gamma * next_value * (1 - terminal_t) - V(s_t)
            A_t     = delta_t + gamma * gae_lambda * (1 - terminal_t) * A_{t+1}

        The ``(1 - terminal_t)`` factor is the **terminal mask**. When step ``t`` is
        a true terminal step, ``terminal_t`` is True (1), so ``(1 - terminal_t)`` is
        0 and it zeroes out *two* things at once: the bootstrap ``next_value`` term
        (the value after a terminal step is treated as 0, Req 5.2) **and** the
        propagated future advantage ``A_{t+1}``. In plain terms, nothing leaks
        across an episode boundary — a new episode's outcomes never flow back into
        the episode that already ended. On a truncation (stored with
        ``terminal=False``) the mask is 1, so bootstrapping *does* continue, which is
        the correct treatment for an episode that was merely cut short.

        Returns a tuple ``(advantages, returns)`` of float32 arrays, each with one
        entry per stored transition in temporal order.
        """

        number_of_steps = self._size

        # Work only over the filled region of the preallocated arrays. Reading the
        # stored fields into clearly named locals keeps the recurrence readable.
        rewards = self._rewards[:number_of_steps]
        values = self._values[:number_of_steps]
        terminals = self._terminals[:number_of_steps]

        # Output arrays, one advantage and one return per stored transition.
        advantages = np.zeros(number_of_steps, dtype=np.float32)

        # The running advantage carried backward from the later step. Beyond the
        # final stored step there is no future advantage yet, so it starts at 0.
        advantage = 0.0

        # Walk from the last stored step back to the first. Each iteration computes
        # one step's advantage using the step immediately after it.
        for t in reversed(range(number_of_steps)):
            # The terminal mask for this step: 0.0 if step t ended an episode,
            # else 1.0. Multiplying by it zeroes both the bootstrap and the
            # carried-back future advantage across an episode boundary.
            not_terminal = 0.0 if terminals[t] else 1.0

            # V(s_{t+1}): the value of the state that follows step t. For the final
            # stored step this comes from bootstrap_value; for every earlier step it
            # is simply the stored value of the next transition.
            if t == number_of_steps - 1:
                next_value = bootstrap_value
            else:
                next_value = values[t + 1]

            # delta_t: the one-step temporal-difference error. The masked next_value
            # drops to 0 after a terminal step, so delta is just reward - V(s_t) there.
            delta = rewards[t] + gamma * next_value * not_terminal - values[t]

            # A_t: the TD error plus the discounted, masked future advantage. After a
            # terminal step the mask zeroes the carried-back advantage, so A_t = delta.
            advantage = delta + gamma * gae_lambda * not_terminal * advantage

            advantages[t] = advantage

        # The return (critic target) for each step is its advantage plus the value
        # the critic originally estimated for that step (Req 5.4).
        returns = advantages + values

        return advantages.astype(np.float32), returns.astype(np.float32)


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Standardize advantages to zero mean / unit variance, with safety guards.

    PPO's policy-gradient step behaves best when the advantages feeding it are
    well-scaled, so when there is more than one advantage we subtract the mean and
    divide by the standard deviation (Req 5.5). Two guards keep this from
    misbehaving on awkward inputs:

    * **Zero / near-zero variance.** If every advantage is (nearly) identical, the
      standard deviation is ~0 and dividing by it would produce infinities or NaNs.
      We add a tiny ``epsilon`` to the denominator so the division is always finite;
      a constant-advantage batch simply maps to ~0 values rather than blowing up.
    * **Single element.** With exactly one advantage there is nothing to standardize
      against (its "spread" is undefined), so we return it unchanged (Req 5.6).

    Returns a new float32 array the same length as ``advantages``.
    """

    advantages = np.asarray(advantages, dtype=np.float32)

    # Single-element (or empty) batch: there is no meaningful spread to normalize
    # against, so return the values unchanged (Req 5.6).
    if advantages.size <= 1:
        return advantages

    # Standardize: subtract the mean, divide by the standard deviation. The small
    # epsilon in the denominator guards against a zero / near-zero std so the result
    # is always finite even when every advantage is identical (Req 5.5).
    mean = advantages.mean()
    std = advantages.std()
    epsilon = 1e-8
    normalized = (advantages - mean) / (std + epsilon)

    return normalized.astype(np.float32)
