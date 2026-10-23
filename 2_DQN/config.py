"""START HERE — all the knobs for the DQN LunarLander agent live in this file.

This module defines a single ``Config`` dataclass that holds every hyperparameter
the agent uses. There is no command-line interface: to change how the agent trains,
you edit a default below (or override it where ``Config`` is constructed in
``main.py``). That keeps the whole experiment configurable from one readable place.

Each field carries a plain-language comment explaining what the knob *does* in
reinforcement-learning terms, so you can tune it with intuition rather than guesswork.

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
    """All hyperparameters for the DQN agent, with documented defaults.

    This is the single place to edit hyperparameters. Any value you leave alone
    keeps the documented default below. Call :meth:`validate` once at startup to
    confirm every value is in range before training begins.
    """

    # --- Core learning hyperparameters -------------------------------------

    # learning_rate: how big a step the optimizer takes when nudging the
    # Q-network toward each learning target. Too high and learning is unstable;
    # too low and it crawls. Must be in (0, 1].
    learning_rate: float = 1e-3

    # gamma (the discount factor): how much the agent values future reward
    # versus immediate reward. 0 means "only care about the reward right now";
    # close to 1 means "value long-term outcomes almost as much as the present".
    # Must be in [0, 1].
    gamma: float = 0.99

    # buffer_capacity: how many past experiences the replay buffer can hold.
    # The agent learns by replaying random past transitions, so a bigger buffer
    # remembers more history. Must be between 1 and 1,000,000.
    buffer_capacity: int = 100_000

    # batch_size: how many past experiences we sample at once for a single
    # learning update. Must be a positive integer, and no larger than the
    # buffer can hold.
    batch_size: int = 64

    # learning_starts: how many experiences must pile up in the buffer before
    # the agent starts learning. We wait so the first learning batches aren't
    # drawn from an almost-empty, highly correlated buffer. Must be between 1
    # and the buffer capacity.
    learning_starts: int = 1_000

    # target_update_interval: how often (in learning-update steps) we refresh
    # the target network by copying the Q-network into it. The target network
    # is a held-still copy that gives us a stable goal to learn toward; syncing
    # it too often removes that stability, too rarely makes it stale.
    # Measured in update steps, must be between 1 and 100,000.
    target_update_interval: int = 1_000

    # --- Exploration schedule (epsilon-greedy) -----------------------------

    # epsilon_start: the exploration rate at the very beginning of training.
    # 1.0 means "act completely at random at first" so the agent sees a wide
    # variety of situations before it trusts its own judgment. In [0, 1].
    epsilon_start: float = 1.0

    # epsilon_min: the floor the exploration rate decays down to. We never stop
    # exploring entirely — a little randomness keeps the agent from getting
    # stuck. Must be in [0, 1] and no larger than epsilon_start.
    epsilon_min: float = 0.05

    # epsilon_decay_steps: over how many interaction steps epsilon slides
    # linearly from epsilon_start down to epsilon_min. Must be a positive
    # integer.
    epsilon_decay_steps: int = 50_000

    # --- Loop length -------------------------------------------------------

    # total_steps: how many total interaction steps with the environment the
    # training loop runs for. More steps means more learning (and more time).
    # Must be between 1 and 100,000,000.
    total_steps: int = 500_000

    # updates_per_step: how many learning updates to perform per interaction
    # step once learning has started. 1 is the classic DQN choice. Must be
    # between 1 and 1,000.
    updates_per_step: int = 1

    # --- Evaluation --------------------------------------------------------

    # eval_episodes: how many episodes the post-training evaluation pass runs
    # with exploration turned off, to measure the trained policy. Must be
    # between 1 and 10,000.
    eval_episodes: int = 10

    # --- Reproducibility and output locations ------------------------------

    # seed: a fixed random seed makes a run reproducible — same seed and
    # hyperparameters produce the same results. Leave as None for a random run;
    # if set, it must be a non-negative integer.
    seed: int | None = None

    # log_dir: where TensorBoard metric logs are written.
    log_dir: str = "artifacts/runs/dqn-lunarlander"

    # video_dir: where recorded evaluation videos are written.
    video_dir: str = "artifacts/videos/dqn-lunarlander"

    # checkpoint_path: file path where the trained Q-network weights are saved.
    checkpoint_path: str = "artifacts/checkpoints/dqn-lunarlander.pt"

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

        # buffer_capacity must be a positive integer in 1..1,000,000.
        if not _is_int(self.buffer_capacity) or not (1 <= self.buffer_capacity <= 1_000_000):
            raise ValueError(
                "buffer_capacity (how many experiences the replay buffer can hold) "
                f"must be a whole number between 1 and 1,000,000, but got {self.buffer_capacity}."
            )

        # batch_size must be a positive integer, no larger than the buffer capacity.
        if not _is_int(self.batch_size) or self.batch_size < 1:
            raise ValueError(
                "batch_size (how many experiences we learn from at once) must be a "
                f"whole number of at least 1, but got {self.batch_size}."
            )
        if self.batch_size > self.buffer_capacity:
            raise ValueError(
                "batch_size (how many experiences we learn from at once) cannot be "
                f"larger than buffer_capacity ({self.buffer_capacity}), but got {self.batch_size}."
            )

        # learning_starts must be a positive integer, no larger than the buffer capacity.
        if not _is_int(self.learning_starts) or not (1 <= self.learning_starts <= self.buffer_capacity):
            raise ValueError(
                "learning_starts (how many experiences to collect before learning "
                "begins) must be a whole number between 1 and buffer_capacity "
                f"({self.buffer_capacity}), but got {self.learning_starts}."
            )

        # target_update_interval must be a positive integer in 1..100,000.
        if not _is_int(self.target_update_interval) or not (1 <= self.target_update_interval <= 100_000):
            raise ValueError(
                "target_update_interval (how often, in update steps, to sync the "
                "target network) must be a whole number between 1 and 100,000, but "
                f"got {self.target_update_interval}."
            )

        # epsilon_start must be in [0, 1].
        if not (0 <= self.epsilon_start <= 1):
            raise ValueError(
                "epsilon_start (the starting exploration rate) must be between 0 and "
                f"1 inclusive, but got {self.epsilon_start}."
            )

        # epsilon_min must be in [0, 1] and no larger than epsilon_start.
        if not (0 <= self.epsilon_min <= 1):
            raise ValueError(
                "epsilon_min (the smallest exploration rate we decay down to) must be "
                f"between 0 and 1 inclusive, but got {self.epsilon_min}."
            )
        if self.epsilon_min > self.epsilon_start:
            raise ValueError(
                "epsilon_min (the exploration floor) cannot be larger than "
                f"epsilon_start ({self.epsilon_start}), but got {self.epsilon_min}."
            )

        # epsilon_decay_steps must be a positive integer.
        if not _is_int(self.epsilon_decay_steps) or self.epsilon_decay_steps < 1:
            raise ValueError(
                "epsilon_decay_steps (how many steps epsilon takes to reach its floor) "
                f"must be a whole number of at least 1, but got {self.epsilon_decay_steps}."
            )

        # total_steps must be a positive integer in 1..100,000,000.
        if not _is_int(self.total_steps) or not (1 <= self.total_steps <= 100_000_000):
            raise ValueError(
                "total_steps (how many interaction steps to train for) must be a whole "
                f"number between 1 and 100,000,000, but got {self.total_steps}."
            )

        # updates_per_step must be a positive integer in 1..1,000.
        if not _is_int(self.updates_per_step) or not (1 <= self.updates_per_step <= 1_000):
            raise ValueError(
                "updates_per_step (how many learning updates per interaction step) must "
                f"be a whole number between 1 and 1,000, but got {self.updates_per_step}."
            )

        # eval_episodes must be a positive integer in 1..10,000.
        if not _is_int(self.eval_episodes) or not (1 <= self.eval_episodes <= 10_000):
            raise ValueError(
                "eval_episodes (how many episodes the evaluation pass runs) must be a "
                f"whole number between 1 and 10,000, but got {self.eval_episodes}."
            )

        # seed is optional, but if provided it must be a non-negative integer.
        if self.seed is not None:
            if not _is_int(self.seed) or self.seed < 0:
                raise ValueError(
                    "seed (the random seed for reproducibility) must be a non-negative "
                    f"whole number when provided, but got {self.seed}."
                )


def _is_int(value: object) -> bool:
    """Return True only for genuine integers, rejecting booleans.

    In Python ``bool`` is a subclass of ``int`` (``True == 1``), so a plain
    ``isinstance(value, int)`` check would accept ``True``/``False`` as valid
    integers. For configuration we want to reject those, so a learner who writes
    ``batch_size=True`` gets a clear error instead of a batch size of 1.
    """
    return isinstance(value, int) and not isinstance(value, bool)
