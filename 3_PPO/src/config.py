"""START HERE — all the knobs for the PPO CarRacing agent live in this file.

This module defines a single ``Config`` dataclass that holds every hyperparameter
the agent uses. There is no command-line interface: to change how the agent trains,
you edit a default below (or override it where ``Config`` is constructed in
``main.py``). That keeps the whole experiment configurable from one readable place.

Each field carries a plain-language comment explaining what the knob *does* in
reinforcement-learning terms, so you can tune it with intuition rather than guesswork.
Where PPO differs from the companion DQN project, the comment says so — most of all
that PPO is *on-policy*: it collects a fresh batch of experience, learns from it once,
and throws it away. There is no replay buffer here.

``Config`` is *frozen* (immutable): once built, its values can't change. That means
the configuration you validate at startup is exactly the configuration that trains
the agent — nothing can quietly mutate it mid-run.

Before anything else happens, call ``config.validate()``. It checks every value
against its sensible range and, on the first bad value, raises a built-in
``ValueError`` that names the offending field in plain language. Catching a bad
config here — before any environment, network, or buffer is created — turns a
mistake into a clear teaching moment instead of a confusing crash deep in training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All hyperparameters for the PPO agent, with documented defaults.

    This is the single place to edit hyperparameters. Any value you leave alone
    keeps the documented default below. Call :meth:`validate` once at startup to
    confirm every value is in range before training begins.
    """

    # --- Core learning hyperparameters -------------------------------------

    # learning_rate: how big a step the optimizer takes when nudging the
    # networks (encoder, actor, and critic) toward each learning target. Too
    # high and learning is unstable; too low and it crawls. Must be in (0, 1].
    learning_rate: float = 3e-4

    # gamma (the discount factor): how much the agent values future reward
    # versus immediate reward. 0 means "only care about the reward right now";
    # close to 1 means "value long-term outcomes almost as much as the present".
    # Must be in [0, 1].
    gamma: float = 0.99

    # gae_lambda: the GAE knob that trades off bias against variance when
    # estimating how much better an action was than expected. Closer to 1 uses
    # more of the actual observed rewards (lower bias, higher variance); closer
    # to 0 leans on the critic's value estimates (higher bias, lower variance).
    # Must be in [0, 1].
    gae_lambda: float = 0.95

    # clip_range: PPO's signature safety rail. It limits how far a single update
    # is allowed to move the policy by clamping the probability ratio to
    # [1 - clip_range, 1 + clip_range]. Small keeps updates cautious. Must be in (0, 1].
    clip_range: float = 0.2

    # update_epochs: how many times we sweep over the freshly collected rollout
    # during one update phase. PPO reuses the same on-policy batch for several
    # passes before discarding it. Must be a whole number between 1 and 1,000.
    update_epochs: int = 10

    # minibatch_size: how many transitions we learn from at once within an
    # epoch. Each epoch partitions the whole rollout into minibatches of this
    # size. Must be a whole number of at least 1 and no larger than rollout_horizon.
    minibatch_size: int = 64

    # rollout_horizon: how many interaction steps we collect into the (single-use)
    # rollout buffer before each update phase. This is the "batch of fresh
    # experience" PPO learns from, then throws away. Must be a whole number
    # between 1 and 100,000.
    rollout_horizon: int = 2048

    # value_coef: how much weight the critic's value loss carries in the combined
    # loss, relative to the policy loss. Must be in [0, 1].
    value_coef: float = 0.5

    # entropy_coef: how strongly we reward the policy for staying random
    # (exploratory). A small nudge keeps the agent from collapsing onto one
    # action too early. Must be in [0, 1].
    entropy_coef: float = 0.01

    # max_grad_norm: a ceiling on the size of the gradient before each optimizer
    # step, which keeps a single rare, huge gradient from destabilizing training.
    # Must be greater than 0.
    max_grad_norm: float = 0.5

    # --- Loop length -------------------------------------------------------

    # total_steps: how many total interaction steps with the environment the
    # training loop runs for. More steps means more learning (and more time).
    # Must be between 1 and 100,000,000.
    total_steps: int = 1_000_000

    # --- Evaluation --------------------------------------------------------

    # eval_episodes: how many episodes the post-training evaluation pass runs
    # with exploration turned off, to measure the trained policy. Must be
    # between 1 and 10,000.
    eval_episodes: int = 5

    # --- Reproducibility and output locations ------------------------------

    # seed: a fixed random seed makes a run reproducible — same seed and
    # hyperparameters produce the same results on the same device. Leave as None
    # for a random run; if set, it must be an integer in [0, 2**32 - 1].
    seed: int | None = None

    # log_dir: where TensorBoard metric logs are written.
    log_dir: str = "artifacts/runs/ppo-car-racing"

    # video_dir: where recorded evaluation videos are written.
    video_dir: str = "artifacts/videos/ppo-car-racing"

    # checkpoint_path: file path where the trained network weights are saved.
    checkpoint_path: str = "artifacts/checkpoints/ppo-car-racing.pt"

    def validate(self) -> None:
        """Check every hyperparameter against its valid range.

        On the first value that is out of range, this raises a built-in
        ``ValueError`` whose message names the offending field in plain
        language. Call this once at startup, before creating any environment,
        network, or buffer, so a bad configuration is caught early with a clear
        explanation rather than a confusing failure mid-training.

        Note that booleans are a subclass of ``int`` in Python, so the integer
        checks below explicitly reject ``bool`` values: a learner who writes
        ``seed=True`` should be told it's not a valid integer, not have ``True``
        silently treated as ``1``.
        """

        # learning_rate must be in (0, 1]: a positive step size, not larger than 1.
        if not (0 < self.learning_rate <= 1):
            raise ValueError(
                "learning_rate (the optimizer step size) must be greater than 0 "
                f"and at most 1, but got {self.learning_rate}."
            )

        # gamma (discount factor) must be in [0, 1].
        if not (0 <= self.gamma <= 1):
            raise ValueError(
                "gamma (the discount factor, how much future reward is valued) "
                f"must be between 0 and 1 inclusive, but got {self.gamma}."
            )

        # gae_lambda must be in [0, 1].
        if not (0 <= self.gae_lambda <= 1):
            raise ValueError(
                "gae_lambda (the GAE bias-variance tradeoff knob) must be between "
                f"0 and 1 inclusive, but got {self.gae_lambda}."
            )

        # clip_range must be in (0, 1]: a positive clip, not larger than 1.
        if not (0 < self.clip_range <= 1):
            raise ValueError(
                "clip_range (how far PPO lets a single update move the policy) must "
                f"be greater than 0 and at most 1, but got {self.clip_range}."
            )

        # entropy_coef must be in [0, 1].
        if not (0 <= self.entropy_coef <= 1):
            raise ValueError(
                "entropy_coef (how strongly we reward exploration) must be between "
                f"0 and 1 inclusive, but got {self.entropy_coef}."
            )

        # value_coef must be in [0, 1].
        if not (0 <= self.value_coef <= 1):
            raise ValueError(
                "value_coef (how much the critic's value loss counts) must be between "
                f"0 and 1 inclusive, but got {self.value_coef}."
            )

        # max_grad_norm must be strictly greater than 0.
        if not (self.max_grad_norm > 0):
            raise ValueError(
                "max_grad_norm (the ceiling on gradient size) must be greater than 0, "
                f"but got {self.max_grad_norm}."
            )

        # rollout_horizon must be a positive integer in 1..100,000.
        if not _is_int(self.rollout_horizon) or not (1 <= self.rollout_horizon <= 100_000):
            raise ValueError(
                "rollout_horizon (how many steps we collect before each update) must "
                f"be a whole number between 1 and 100,000, but got {self.rollout_horizon}."
            )

        # update_epochs must be a positive integer in 1..1,000.
        if not _is_int(self.update_epochs) or not (1 <= self.update_epochs <= 1_000):
            raise ValueError(
                "update_epochs (how many passes over the rollout each update does) must "
                f"be a whole number between 1 and 1,000, but got {self.update_epochs}."
            )

        # minibatch_size must be a positive integer, no larger than the rollout horizon.
        if not _is_int(self.minibatch_size) or self.minibatch_size < 1:
            raise ValueError(
                "minibatch_size (how many transitions we learn from at once) must be a "
                f"whole number of at least 1, but got {self.minibatch_size}."
            )
        if self.minibatch_size > self.rollout_horizon:
            raise ValueError(
                "minibatch_size (how many transitions we learn from at once) cannot be "
                f"larger than rollout_horizon ({self.rollout_horizon}), but got "
                f"{self.minibatch_size}."
            )

        # total_steps must be a positive integer in 1..100,000,000.
        if not _is_int(self.total_steps) or not (1 <= self.total_steps <= 100_000_000):
            raise ValueError(
                "total_steps (how many interaction steps to train for) must be a whole "
                f"number between 1 and 100,000,000, but got {self.total_steps}."
            )

        # eval_episodes must be a positive integer in 1..10,000.
        if not _is_int(self.eval_episodes) or not (1 <= self.eval_episodes <= 10_000):
            raise ValueError(
                "eval_episodes (how many episodes the evaluation pass runs) must be a "
                f"whole number between 1 and 10,000, but got {self.eval_episodes}."
            )

        # seed is optional, but if provided it must be an integer in [0, 2**32 - 1].
        if self.seed is not None:
            if not _is_int(self.seed) or not (0 <= self.seed <= 2**32 - 1):
                raise ValueError(
                    "seed (the random seed for reproducibility) must be a whole number "
                    f"between 0 and 2**32 - 1 when provided, but got {self.seed}."
                )


def _is_int(value: object) -> bool:
    """Return True only for genuine integers, rejecting booleans.

    In Python ``bool`` is a subclass of ``int`` (``True == 1``), so a plain
    ``isinstance(value, int)`` check would accept ``True``/``False`` as valid
    integers. For configuration we want to reject those, so a learner who writes
    ``minibatch_size=True`` gets a clear error instead of a minibatch size of 1.
    """
    return isinstance(value, int) and not isinstance(value, bool)
