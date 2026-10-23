"""The temporal-difference (TD) target — the conceptual heart of DQN.

Everything else in this project exists to support one question: *what number
should the Q-network have predicted?* That number is the **TD target**, and this
module computes it.

Recall what a Q-value means: ``Q(state, action)`` is the agent's estimate of the
total future reward it expects to collect if it takes ``action`` in ``state`` and
then keeps acting well afterward. Training nudges the network's prediction for the
action it actually took toward a better-informed estimate of that same quantity —
the TD target. The target is "better informed" because it gets to peek one real
step into the future: it uses the reward the environment actually handed back,
plus an estimate of how good the *next* state is.

There are two cases, and they are the whole story:

* **The episode did NOT end on this step** (``terminated`` is false). There is a
  future to account for, so the target is::

      reward  +  gamma * (best Q-value the target network expects in next_state)

  In words: *the reward we just got now, plus the discounted best future value
  the target network expects from where we landed.* ``gamma`` (the discount
  factor) shrinks the future term because reward later is worth a little less
  than reward now. We take the **maximum** over next-state actions because from
  ``next_state`` the agent is free to pick its best action — DQN assumes optimal
  play thereafter.

* **The episode DID end on this step** (``terminated`` is true). There is no
  future after a terminal state — the lander has crashed or landed, the episode
  is over — so the target is **just the reward**. We deliberately drop the
  future term entirely; adding a discounted next-state value here would be
  inventing reward for a future that does not exist.

Why the *target* network (and not the online Q-network) for the next-state value?
This is the stabilizing trick introduced in ``networks.py`` finally paying off.
If we computed the next-state value with the very network we are training, the
target would shift every time we took a gradient step — we would be chasing a
moving goal, and learning would wobble or diverge. Instead we read the next-state
value from the *target* network, the periodically-synced frozen snapshot, so the
goal stays still while the online network learns toward it.

The function below is written for a whole :class:`~replay_buffer.Batch` of
transitions at once (that is how the trainer uses it), but it reads exactly like
the two-case story above, just applied elementwise across the batch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from networks import QNetwork


def compute_td_target(
    target_network: QNetwork,
    rewards: Tensor,
    next_states: Tensor,
    terminateds: Tensor,
    gamma: float,
) -> Tensor:
    """Compute the TD target for a batch of transitions.

    For each transition in the batch this returns the number the online
    Q-network's prediction should be trained toward:

    * ``reward + gamma * max_a Q_target(next_state, a)`` when the episode did not
      terminate on this step (there is a discounted future to account for), and
    * ``reward`` alone when it did terminate (there is no future after the end).

    The next-state value comes from ``target_network`` — the frozen, periodically
    synced copy of the Q-network — rather than the online network being trained,
    so the learning goal stays still while we chase it (see the module docstring
    and ``networks.py`` for why this matters).

    Parameters
    ----------
    target_network:
        The target network (the frozen snapshot) used to estimate the value of
        each ``next_state``.
    rewards:
        Shape ``(batch,)``. The reward received on each transition's step.
    next_states:
        Shape ``(batch, obs_dim)``. The observation the agent landed in after
        each step.
    terminateds:
        Shape ``(batch,)``. True where the episode ended on this step, false
        otherwise. May be a bool tensor; it is used as a mask below.
    gamma:
        The discount factor in ``[0, 1]`` that shrinks the future term.

    Returns
    -------
    Tensor
        Shape ``(batch,)``. The TD target for each transition. This value is a
        fixed learning goal, so it carries no gradient (see ``torch.no_grad``
        below) — we train the online network toward it, we do not train it.
    """

    # The TD target is a goal we aim the online network at, never something we
    # backpropagate through. Wrapping the whole computation in no_grad keeps the
    # target network out of the gradient graph: it supplies a fixed number, full
    # stop.
    with torch.no_grad():
        # Ask the target network for its Q-value estimates in each next_state.
        # Shape: (batch, n_actions) — one estimated value per action per sample.
        next_state_q_values = target_network(next_states)

        # From each next_state the agent is free to choose its best action, so we
        # take the maximum Q-value over the action dimension. ``.values`` discards
        # the argmax indices (we only care about the value, not which action).
        # Shape: (batch,) — the best future value the target network expects.
        best_next_value = next_state_q_values.max(dim=1).values

        # Discount that future value, because reward collected later is worth a
        # little less than reward collected now.
        discounted_future = gamma * best_next_value

        # --- The two cases, applied elementwise across the batch -------------
        #
        # Non-terminal transitions (terminated == False): reward now, plus the
        # discounted best future value the target network expects. We express the
        # "is there a future?" switch as the factor ``(1 - terminated)``: it is
        # 1.0 when the episode continued (keep the future term) and 0.0 when it
        # ended (drop the future term).
        #
        # Terminal transitions (terminated == True): just the reward, because
        # there is no future after the episode ends — the (1 - terminated) factor
        # zeroes the future term out entirely.
        episode_continues = (~terminateds.bool()).float()
        td_target = rewards + episode_continues * discounted_future

    return td_target
