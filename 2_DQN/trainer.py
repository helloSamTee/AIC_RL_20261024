"""The online training loop — the capstone that ties every DQN piece together.

This is the file to read *last*. By now you have met each building block on its
own: the :class:`~config.Config` of knobs, the :class:`~networks.QNetwork` brain
and its frozen target-network twin, the :class:`~replay_buffer.ReplayBuffer` of
past experience, the epsilon-greedy :mod:`policy`, the :func:`~td_target.compute_td_target`
math, the :func:`~checkpoint.save_checkpoint` snapshot, and the
:class:`~logger.TensorBoardLogger`. The :class:`Trainer` below is where they all
meet and start working together.

The heart of it is :meth:`Trainer.train`, written as a plain narration of one
loop iteration:

    select an action  ->  step the environment  ->  store the transition
      ->  (once warmed up) sample a batch and learn
      ->  sync the target network on schedule
      ->  decay epsilon  ->  log

Each of those steps is a single call to a well-named helper, so the body of the
loop reads almost exactly like that list. The one place with real arithmetic —
turning a sampled batch into a gradient step — is pulled out into its own small
method, :meth:`Trainer._learn_from_batch`, so the loop itself stays short and the
TD-target math lives in one focused place.

This is the **online, off-policy** DQN algorithm: interaction (filling the buffer
with fresh experience) and learning (replaying random past experience) happen
together in a single loop, not in two separate phases.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint import save_checkpoint
from config import Config
from env_factory import create_env
from logger import TensorBoardLogger
from networks import QNetwork, hard_update
from policy import epsilon_at, select_action
from replay_buffer import Batch, ReplayBuffer
from td_target import compute_td_target


class Trainer:
    """Runs the online DQN training loop that ties every component together.

    Construct a ``Trainer`` from a validated :class:`~config.Config` and then call
    :meth:`train`. The constructor builds everything the loop needs — the
    environment, the online Q-network and its frozen target twin, the optimizer,
    the replay buffer, a reproducible random generator, and the TensorBoard
    logger — wiring the pieces together in one readable place. :meth:`train` then
    runs the interaction-and-learning loop and saves a checkpoint when it is done.

    The ``Trainer`` deliberately owns its collaborators (rather than being handed
    them) so a learner can open this one file and see, top to bottom, exactly how
    a DQN agent is assembled and run.
    """

    def __init__(self, cfg: Config) -> None:
        # Validate the configuration before building anything. A bad knob should
        # fail here, with a clear message naming the field, rather than deep
        # inside the loop after resources have been created.
        cfg.validate()
        self.cfg = cfg

        # --- Reproducibility: seed every source of randomness ---------------
        # DQN draws on three independent random streams: the environment's own
        # randomness, the action-selection / buffer-sampling randomness, and the
        # network weight initialization. To make a run reproducible we must seed
        # all three from the single configured seed (Requirement 10.2). When
        # cfg.seed is None we skip seeding and let each source pick its own, which
        # gives an intentionally non-reproducible run.
        #
        # We keep a dedicated numpy Generator (self.rng) for the agent's own
        # choices — action selection and buffer sampling — and pass it explicitly
        # everywhere randomness is needed, so there is no hidden global state.
        self.rng = self._seed_everything(cfg.seed)

        # --- Environment -----------------------------------------------------
        # env_factory isolates all the Gymnasium setup; here we just ask for a
        # seeded LunarLander. We read the observation size and action count off
        # the environment rather than hard-coding 8 and 4, so the networks are
        # sized from the environment itself.
        self.env = create_env(seed=cfg.seed)
        obs_dim = self.env.observation_space.shape[0]
        self.n_actions = int(self.env.action_space.n)

        # --- Networks --------------------------------------------------------
        # The online Q-network is the one we train. The target network is a
        # second copy with the SAME architecture, held fixed between syncs to
        # give the learning target something stable to aim at.
        self.q_network = QNetwork(obs_dim, self.n_actions)
        self.target_network = QNetwork(obs_dim, self.n_actions)

        # Initialize the target network identical to the Q-network before the
        # first update, so the very first learning target is well-defined
        # (Requirement 2.3). This single call *is* the target-network sync.
        hard_update(self.target_network, self.q_network)

        # The target network is never trained directly — it only supplies
        # next-state value estimates — so put it in eval mode. (Gradients are
        # already kept off it inside compute_td_target's no_grad block.)
        self.target_network.eval()

        # --- Optimizer -------------------------------------------------------
        # Adam adjusts the Q-network's weights to reduce the loss. It optimizes
        # only the online network's parameters; the target network is left alone.
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=cfg.learning_rate)

        # --- Replay buffer ---------------------------------------------------
        # The store of past experience we sample random learning batches from.
        self.buffer = ReplayBuffer(cfg.buffer_capacity)

        # --- Logger ----------------------------------------------------------
        # Writes episode reward, loss, and epsilon to TensorBoard. It fails soft:
        # if the log directory can't be created, logging disables itself and
        # training continues.
        self.logger = TensorBoardLogger(cfg.log_dir)

        # --- Counters the loop maintains ------------------------------------
        # episode_index counts completed episodes (starts at 0, Requirement 6.1).
        # update_step counts completed Q-network updates (starts at 0,
        # Requirement 6.2) and also drives the target-sync schedule.
        self.episode_index = 0
        self.update_step = 0

    def _seed_everything(self, seed: int | None) -> np.random.Generator:
        """Seed Python, NumPy, and PyTorch for reproducibility; return the agent's RNG.

        When ``seed`` is provided we seed every random source DQN relies on so
        two runs with the same seed and hyperparameters behave identically
        (Requirement 10.2):

        * ``random`` and ``torch`` are seeded globally — torch in particular
          governs the network's random weight initialization.
        * We create and return a dedicated :class:`numpy.random.Generator` seeded
          from the same number. This is the generator the agent threads through
          ``select_action`` and ``buffer.sample``, so the agent's own choices are
          reproducible without leaning on NumPy's hidden global state.

        When ``seed`` is ``None`` we seed nothing and return an unseeded
        generator, giving a deliberately non-reproducible run.
        """
        if seed is None:
            return np.random.default_rng()

        random.seed(seed)
        torch.manual_seed(seed)
        return np.random.default_rng(seed)

    def _select_action_for(self, obs: np.ndarray, epsilon: float) -> int:
        """Choose one action for the current observation with the epsilon-greedy rule.

        We run the observation through the *online* Q-network to get its per-action
        value estimates, then hand those to :func:`policy.select_action`, which
        either explores (random action, with probability ``epsilon``) or exploits
        (the greedy best action). The network expects a batch, so we add a batch
        dimension of 1 before the forward pass and read the single row back out.
        """
        # Convert the observation to a float tensor with a leading batch dim of 1:
        # shape (obs_dim,) -> (1, obs_dim), because the network always works on
        # batches. no_grad because selecting an action is not something we learn
        # through — we only learn in _learn_from_batch.
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor).squeeze(0)

        # The policy flips the epsilon-weighted coin and returns a valid action
        # index. We pass our seeded rng so the exploration choices are reproducible.
        return select_action(q_values, epsilon, self.n_actions, self.rng)

    def _learn_from_batch(self, batch: Batch) -> float:
        """Perform one gradient update of the Q-network on a sampled batch.

        This is the one place the TD-target math actually happens, so it is worth
        reading slowly. Given a batch of past transitions, a single learning step
        is:

        1. **Compute the TD target** — the number each prediction *should* have
           been. For a transition that did not end the episode this is
           ``reward + gamma * max_a Q_target(next_state, a)``; for one that did
           end it is just ``reward``. The next-state value comes from the frozen
           *target* network (not the online one) so the goal stays still while we
           chase it. :func:`td_target.compute_td_target` does this elementwise for
           the whole batch and already shields it from gradients.
        2. **Compute the current prediction** — run the batch of states through
           the *online* Q-network and pick out the Q-value for the action the
           agent actually took in each transition (a ``gather`` along the action
           axis). That is the number we are trying to improve.
        3. **Measure the error and step** — compare prediction to target with a
           smooth-L1 (Huber) loss, which behaves like mean-squared error for small
           errors but is less sensitive to the occasional large outlier, then run
           one optimizer step to nudge the online network's weights toward the
           target.

        Returns the scalar loss value for logging.
        """
        # Move the batch's NumPy arrays into torch tensors with the dtypes the
        # network and loss expect. states/next_states/rewards are floats; actions
        # are integer indices; terminated is a boolean mask.
        states = torch.as_tensor(batch.states, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32)
        terminateds = torch.as_tensor(batch.terminateds, dtype=torch.bool)

        # Step 1: the learning goal, computed from the frozen target network.
        # Shape (batch,).
        td_target = compute_td_target(
            self.target_network, rewards, next_states, terminateds, self.cfg.gamma
        )

        # Step 2: the online network's current estimate for the action actually
        # taken. The forward pass gives a Q-value for every action, shape
        # (batch, n_actions); we gather the column matching each taken action to
        # get one predicted value per transition, shape (batch,).
        all_q_values = self.q_network(states)
        taken_action_q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Step 3: Huber (smooth-L1) loss between prediction and target, then one
        # optimizer step. We zero the previous gradients, backpropagate the loss,
        # and let Adam nudge the online weights.
        loss = F.smooth_l1_loss(taken_action_q_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train(self) -> None:
        """Run the full online training loop, then save a checkpoint.

        One iteration of the loop reads as a short story:

            select an action -> step the environment -> store the transition
              -> (once warmed up) sample batches and learn
              -> sync the target network on schedule
              -> decay epsilon -> log

        The loop runs for ``cfg.total_steps`` interaction steps total
        (Requirement 5.1). Learning only begins once the buffer has accumulated
        ``learning_starts`` transitions (Requirements 5.2, 5.3). When an episode
        ends we log its total reward, reset the environment, and start the next
        one (Requirements 5.6, 6.1). When all the steps are done we persist the
        trained Q-network (Requirement 8.1). This method does NOT run the
        evaluation pass — that is a separate phase.
        """
        # Begin the first episode. We track the running total reward so we can log
        # it when the episode ends.
        obs, _info = self.env.reset(seed=self.cfg.seed)
        episode_reward = 0.0

        for global_step in range(self.cfg.total_steps):
            # How much to explore right now, from the linear epsilon schedule.
            epsilon = epsilon_at(global_step, self.cfg)

            # --- Interact: act, step, store --------------------------------
            action = self._select_action_for(obs, epsilon)
            next_obs, reward, terminated, truncated, _info = self.env.step(action)

            # Store the transition for later replay. Note we store `terminated`
            # (the episode genuinely ended) and NOT `truncated` (a time-limit cut
            # off): only a true terminal state means "there is no future" for the
            # TD target, so a truncation must not zero out the future value.
            self.buffer.add(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_reward += float(reward)

            # --- Learn: only once the buffer has warmed up -----------------
            # We wait for `learning_starts` transitions before any update so the
            # first learning batches are drawn from a reasonably sized, less
            # correlated buffer rather than a nearly empty one.
            if len(self.buffer) >= self.cfg.learning_starts:
                # Perform the configured number of updates for this interaction
                # step. Each update samples its own fresh random batch.
                for _ in range(self.cfg.updates_per_step):
                    batch = self.buffer.sample(self.cfg.batch_size, self.rng)
                    # sample() returns None only if the buffer somehow holds fewer
                    # than batch_size transitions; learning_starts >= batch_size in
                    # any sane config, but we guard anyway and simply skip.
                    if batch is None:
                        break

                    loss = self._learn_from_batch(batch)

                    # Log this update's loss and the current epsilon against the
                    # update-step index (the logger records epsilon only when it
                    # has changed). Then advance the update counter.
                    self.logger.log_loss(self.update_step, loss)
                    self.logger.log_epsilon(self.update_step, epsilon)

                    # --- Sync the target network on schedule ----------------
                    # We refresh the frozen target only every
                    # `target_update_interval` UPDATE steps, not every step: the
                    # whole point of the target network is to hold still between
                    # syncs so the learning goal doesn't move while we chase it.
                    # Syncing too often would throw that stability away.
                    self.update_step += 1
                    if self.update_step % self.cfg.target_update_interval == 0:
                        hard_update(self.target_network, self.q_network)

            # --- Handle episode end ---------------------------------------
            # An episode ends on either a true terminal state or a truncation
            # (time limit). Either way we log the episode's total reward, reset
            # the environment, and begin a fresh episode on the next iteration.
            if terminated or truncated:
                self.logger.log_episode_reward(self.episode_index, episode_reward)
                self.episode_index += 1

                obs, _info = self.env.reset()
                episode_reward = 0.0

        # --- Training complete: persist the trained network ----------------
        # Save a snapshot of the online Q-network's weights so the policy can be
        # reused (e.g. by the evaluation pass) without retraining.
        save_checkpoint(self.q_network, self.cfg.checkpoint_path)
