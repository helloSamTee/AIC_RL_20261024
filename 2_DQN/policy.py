"""The epsilon-greedy policy and its exploration schedule — "how does the agent explore?"

A learning agent faces a tension on every single step, the **explore-vs-exploit
trade-off**:

* *Exploit* — take the action the agent currently *believes* is best (the one
  with the highest estimated Q-value). This cashes in on what it has learned.
* *Explore* — take a random action instead, to discover outcomes the agent has
  not tried yet. Early on its beliefs are mostly guesses, so blindly exploiting
  them would trap it in whatever looked good first.

An agent that only exploits never learns about better options it hasn't sampled;
an agent that only explores never puts its knowledge to use. **Epsilon-greedy**
is the classic, simple compromise: with probability ``epsilon`` take a random
action (explore), and the rest of the time take the greedy best-known action
(exploit). ``epsilon`` is just the "chance of acting randomly."

The trick is that the *right* amount of exploration changes over training. At the
start the agent knows nothing, so we want ``epsilon`` high (explore a lot). As it
learns, we want ``epsilon`` to shrink toward a small floor so the agent mostly
exploits what it now knows — but we never drop it all the way to zero, because a
little lingering randomness keeps the agent from getting permanently stuck.

This module keeps those two ideas in two separate functions *on purpose*:

* :func:`epsilon_at` answers **"how much should we explore right now?"** — it maps
  the current training step to an exploration rate via a linear decay schedule.
* :func:`select_action` answers **"given that, what action do we take?"** — it
  flips the epsilon-weighted coin and returns either a random action (explore) or
  the greedy best action (exploit).

Splitting them keeps the decay schedule easy to reason about (and easy to graph
on its own) and keeps the per-step choice down to a single readable branch.
"""

from __future__ import annotations

import numpy as np

from config import Config


def epsilon_at(step: int, cfg: Config) -> float:
    """Return the exploration rate for a given interaction ``step``.

    Epsilon follows a *linear decay schedule*: it starts high at
    ``cfg.epsilon_start`` (explore a lot, because the agent knows nothing yet)
    and slides down in a straight line to the small floor ``cfg.epsilon_min``
    (mostly exploit what has been learned) over the course of
    ``cfg.epsilon_decay_steps`` steps. After that it stays pinned at the floor —
    we keep a little randomness forever so the agent never gets permanently
    stuck, but we never explore heavily again.

    The result is monotonically non-increasing in ``step`` and always lies within
    ``[cfg.epsilon_min, cfg.epsilon_start]``, including for odd inputs like a
    negative step or a step well past the decay horizon (the final clamp below
    guarantees this).
    """

    # How far we are through the decay window, as a fraction in principle from
    # 0.0 (very start) to 1.0 (decay complete). We compute it plainly here and
    # let it run outside [0, 1] for out-of-range steps — the clamp at the end
    # cleans that up, so we don't need to special-case negative or large steps.
    fraction_elapsed = step / cfg.epsilon_decay_steps

    # Linear interpolation from epsilon_start down to epsilon_min. At
    # fraction_elapsed = 0 this is epsilon_start; at 1 it is epsilon_min.
    total_drop = cfg.epsilon_start - cfg.epsilon_min
    interpolated_epsilon = cfg.epsilon_start - fraction_elapsed * total_drop

    # Final clamp to the valid band [epsilon_min, epsilon_start]. This both pins
    # epsilon to the floor once decay is done and guards against out-of-range
    # steps (a negative step would interpolate above epsilon_start; a very large
    # step would push below epsilon_min).
    clamped_epsilon = min(cfg.epsilon_start, max(cfg.epsilon_min, interpolated_epsilon))
    return clamped_epsilon


def select_action(
    q_values,
    epsilon: float,
    n_actions: int,
    rng: np.random.Generator,
    greedy: bool = False,
) -> int:
    """Choose an action with the epsilon-greedy rule, returning an index in ``[0, n_actions)``.

    This is the per-step explore-vs-exploit decision:

    * With probability ``epsilon`` we **explore** — return a uniformly random
      action index.
    * Otherwise we **exploit** — return the greedy action, the ``argmax`` over
      ``q_values``.

    Set ``greedy=True`` to turn off exploration entirely (used by the evaluation
    pass, which measures the trained policy without any random actions): it always
    returns the greedy action and never samples randomly, regardless of
    ``epsilon``.

    Tie-breaking: when two or more actions share the top Q-value, we break the tie
    toward the **lowest** action index. ``numpy.argmax`` already returns the first
    (lowest) index of the maximum, so the behavior is deliberate, not accidental.

    ``q_values`` may be a NumPy array or a PyTorch tensor; we convert it to a
    NumPy array before taking the argmax so both work the same way. The return
    value is always a valid index in ``[0, n_actions)``.
    """

    # --- EXPLORE branch --------------------------------------------------
    # Draw a uniform float in [0, 1). When it lands below epsilon we act
    # randomly. (When greedy=True we skip this entirely and always exploit.)
    if not greedy:
        exploration_roll = rng.random()
        if exploration_roll < epsilon:
            # Pick a uniformly random action index in [0, n_actions).
            random_action = int(rng.integers(n_actions))
            return random_action

    # --- EXPLOIT branch --------------------------------------------------
    # Take the greedy best-known action. Convert q_values to a NumPy array first
    # so this works whether the caller passed a NumPy array or a torch tensor.
    q_values_array = np.asarray(q_values)

    # argmax returns the first (lowest) index of the maximum value, which is
    # exactly the lowest-index tie-breaking we want when several actions tie.
    greedy_action = int(np.argmax(q_values_array))
    return greedy_action
