"""The Q-network and the target-network sync helper — "what is a target network?"

This module holds two things a first-time DQN reader needs to meet together:

1. ``QNetwork`` — the neural network that *is* the agent's brain. Given an
   observation (what the agent currently sees), it estimates a Q-value for every
   action: "how good do I expect things to turn out if I take this action now?"
   The agent picks actions by looking at these estimates, and training nudges
   them to be more accurate over time.

2. ``hard_update`` — the one-line operation that keeps a *second* copy of the
   network, the **target network**, in sync with the main Q-network.

Why a second network at all? During training we improve the Q-network by pushing
its estimates toward a learning target that is itself computed from a network's
estimates of the next state. If we used the *same* network to both make a guess
and define the goal for that guess, the goal would shift every time we learned —
like trying to hit a target that moves the instant you aim at it. That feedback
loop makes learning wobble or diverge.

The fix is simple: keep a second copy of the Q-network, the **target network**,
and hold its weights *fixed* while we train the main network against it. Every so
often we copy the main network's weights into the target (that copy is
``hard_update``). Between those copies the target stands still, so the learning
goal stays put long enough for the main network to actually chase it. The target
network is nothing more exotic than "a snapshot of the Q-network from a little
while ago."
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QNetwork(nn.Module):
    """A small multilayer perceptron (MLP) that estimates Q-values per action.

    The network maps an observation vector of length ``obs_dim`` to a vector of
    length ``n_actions`` — one estimated Q-value for each discrete action the
    agent can take. For LunarLander that is 8 numbers in and 4 numbers out.

    The architecture is deliberately plain: a couple of fully connected hidden
    layers with ReLU activations, then a final linear layer that outputs one
    value per action. The ``hidden`` argument lists the width of each hidden
    layer, so ``hidden=(128, 128)`` builds two hidden layers of 128 units each.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        
        layers = []
        input_width = obs_dim

        # Build the hidden layers 
        # Each hidden layer = A Linear layer followed by a ReLU activation.
        for layer_width in hidden:
            layers.append(nn.Linear(input_width, layer_width))
            # ReLU just keeps positive values and zeros
            # out negatives, which is the standard nonlinearity between hidden layers.
            layers.append(nn.ReLU())
            input_width = layer_width

        # The final layer has no activation: it emits the raw Q-value estimates,
        # one per action. Its input width is the width of the last hidden layer.
        layers.append(nn.Linear(input_width, n_actions))
        
        self.net = nn.Sequential(*layers)

    def forward(self, obs: Tensor) -> Tensor:
        """Run an observation (or a batch of them) through the network.

        ``obs`` has shape ``(batch, obs_dim)``. The return value has shape
        ``(batch, n_actions)`` — one Q-value per action for each observation in
        the batch.
        """

        # Raw observation -> hidden layer 1 -> hidden layer 2 -> output layer
        return self.net(obs)


def hard_update(target: QNetwork, source: QNetwork) -> None:
    """Copy every parameter from ``source`` into ``target`` (the target-network sync).

    This single operation *is* the target-network update. After it runs, every
    weight and bias in ``target`` is an exact, elementwise copy of the
    corresponding value in ``source``, so the two networks are momentarily
    identical.

    WHEN it runs:
      * Once at the very start of training, to initialize the target network as
        an exact copy of the freshly created Q-network (so the first learning
        target is well-defined).
      * Again every ``target_update_interval`` update steps thereafter, to
        refresh the frozen target with the latest Q-network weights.

    WHY it runs: between these copies the target network is held fixed, giving
    the Q-network a stable goal to learn toward. Syncing periodically lets the
    target slowly catch up to the improving Q-network without the moving-target
    instability that would arise from using one network for both roles.

    ``source`` is the online Q-network being trained; ``target`` is the frozen
    copy that supplies next-state value estimates during learning.
    """

    # torch.no_grad() because copying weights is bookkeeping, not something we
    # want the autograd engine to track or build gradients through.
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.copy_(source_param)
