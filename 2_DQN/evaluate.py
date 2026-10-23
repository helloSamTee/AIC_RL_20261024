"""The post-training evaluation pass — "how good did the trained agent actually get?"

Training runs with exploration turned *on*: the epsilon-greedy policy keeps taking
random actions so the agent discovers the environment. That makes the training
reward curve a noisy, deliberately-hampered picture of the agent's skill — it is
partly measuring the random actions, not the learned policy.

This module measures the policy *properly*, by turning exploration all the way
**off**. It reloads the trained Q-network from its checkpoint and runs it for a
handful of episodes in which every single action is the greedy best-known action
(the one with the highest Q-value), never a random one. The mean total reward
across those episodes is the honest, exploration-free score for the trained agent.

Two things make this pass worth its own file:

* **Greedy actions only.** We reuse :func:`policy.select_action` with
  ``greedy=True`` — the single flag that disables exploration. A reader can see
  that the *only* difference from the training loop's action selection is that one
  flag: same network, same argmax, just no random coin-flip.
* **A recorded video.** We run in a video-recording environment so at least one
  episode of the trained agent is saved to disk as an ``.mp4``. Numbers tell you
  the agent scored well; the video lets you actually *watch* it fly the lander,
  which is the most convincing evidence that learning worked.

Recording to disk can fail (a full disk, a missing codec, a bad path). Because the
point of this pass is to *measure* the policy, a video-writing problem must not
throw away the measurement: we catch such a failure, surface a plain-language
error that names the problem, and let the evaluation pass finish and return its
mean reward anyway.
"""

from __future__ import annotations

import numpy as np
import torch

from config import Config
from env_factory import create_recording_env
from logger import TensorBoardLogger
from networks import QNetwork
from checkpoint import load_checkpoint
from policy import select_action


def evaluate(cfg: Config, logger: TensorBoardLogger) -> float:
    """Run the greedy policy for ``cfg.eval_episodes`` episodes and return the mean total reward.

    Steps, in order:

    1. Build a video-recording LunarLander environment via
       :func:`env_factory.create_recording_env`, so at least one episode of the
       trained agent is saved to disk (Requirement 7.5). We read the observation
       size and action count off the environment rather than hard-coding them.
    2. Construct a :class:`~networks.QNetwork` of the right shape and restore the
       trained weights from ``cfg.checkpoint_path`` with
       :func:`checkpoint.load_checkpoint`. If the checkpoint is missing, the
       natural ``FileNotFoundError`` surfaces untouched (its message names the
       path). Put the network in eval mode.
    3. Run ``cfg.eval_episodes`` episodes (Requirement 9.2). On every step we pick
       the action **greedily** — :func:`policy.select_action` with
       ``greedy=True`` always returns the argmax Q-value and never a random action
       (Requirement 9.1) — and accumulate each episode's total reward.
    4. Compute ``mean_reward`` as the arithmetic mean of the per-episode total
       rewards (Requirement 9.4) and log it as a single scalar (Requirement 9.5).

    A failure while writing a video does not abort the pass (Requirement 7.4):
    ``RecordVideo`` writes frames when an episode ends and when the env is closed,
    so we wrap the episode loop and the close in a guard that catches an
    ``OSError``, prints an error naming the problem, and lets the pass finish.
    """

    # --- Step 1: build the video-recording environment -------------------
    # The recorder writes at least one episode to disk so we can watch the
    # trained agent. We seed it for a reproducible evaluation run.
    env = create_recording_env(video_dir=cfg.video_dir, seed=cfg.seed)

    # Size the network from the environment itself (8 obs, 4 actions for
    # LunarLander) instead of hard-coding those numbers.
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    # --- Step 2: rebuild the network and load the trained weights ---------
    # Construct an identically-shaped Q-network, then restore the checkpointed
    # parameters in place. A missing checkpoint raises FileNotFoundError, which
    # we let surface because its message names the path a learner mistyped.
    q_network = QNetwork(obs_dim, n_actions)
    load_checkpoint(q_network, cfg.checkpoint_path)

    # eval mode: this network is only being read from, never trained here.
    q_network.eval()

    # A seeded RNG satisfies select_action's signature. With greedy=True it is
    # never actually consulted (no random actions), but we pass a real generator
    # so action selection stays reproducible and the interface is honored.
    rng = np.random.default_rng(cfg.seed)

    # We collect each episode's total reward so we can average them at the end.
    episode_total_rewards: list[float] = []

    # --- Steps 3 & 4: run greedy episodes, guarding video writes ----------
    # RecordVideo flushes video to disk when an episode ends and when the env is
    # closed. We wrap the whole run (and the close) so a video-write OSError is
    # surfaced but does not abort the measurement (Requirement 7.4).
    try:
        for _episode_index in range(cfg.eval_episodes):
            obs, _info = env.reset(seed=cfg.seed)
            done = False
            episode_reward = 0.0

            # Play one full episode with no exploration at all.
            while not done:
                # Shape (obs_dim,) -> (1, obs_dim): the network always works on a
                # batch, so add a leading batch dimension, run the forward pass
                # under no_grad (we are not learning), then squeeze it back out.
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_network(obs_tensor).squeeze(0)

                # greedy=True is the single flag that turns exploration OFF: this
                # always returns the argmax action, never a random one. epsilon is
                # irrelevant in greedy mode, so we pass 0.0 to make that explicit.
                action = select_action(
                    q_values, epsilon=0.0, n_actions=n_actions, rng=rng, greedy=True
                )

                obs, reward, terminated, truncated, _info = env.step(action)
                episode_reward += float(reward)

                # An episode ends on a true terminal state or a time-limit
                # truncation; either way this episode is over.
                done = terminated or truncated

            episode_total_rewards.append(episode_reward)
    except OSError as error:
        # A video-write failure must not throw away the evaluation. Surface a
        # plain-language error naming the problem and continue to the averaging
        # and logging below with whatever episodes completed.
        print(
            f"[evaluate] Failed to write evaluation video to '{cfg.video_dir}' "
            f"({error}). Continuing the evaluation pass without aborting."
        )
    finally:
        # Closing flushes the final video. Guard it too, because the flush itself
        # can be what fails — again we surface the error rather than abort.
        try:
            env.close()
        except OSError as error:
            print(
                f"[evaluate] Failed to finalize evaluation video in "
                f"'{cfg.video_dir}' ({error}). Continuing without aborting."
            )

    # The arithmetic mean of the per-episode total rewards is the trained
    # policy's exploration-free score. (eval_episodes is validated to be >= 1, so
    # there is always at least one episode to average; we still guard against an
    # empty list defensively in case a video failure cut the run short.)
    mean_reward = (
        float(np.mean(episode_total_rewards)) if episode_total_rewards else 0.0
    )

    # Log the single summary scalar for this run (Requirement 9.5).
    logger.log_eval_mean_reward(mean_reward)

    return mean_reward
