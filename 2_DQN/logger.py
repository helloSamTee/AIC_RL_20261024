"""Writing training metrics to TensorBoard — "how do I watch the agent learn?"

Training a DQN agent is a slow, mostly-silent process: numbers shuffle around
inside the network for hundreds of thousands of steps. This module gives that
process a visible heartbeat. It records a handful of scalar metrics to a
TensorBoard log directory so a learner can open TensorBoard and literally watch
the curves move while training runs.

Three metrics tell most of the story:

* **Episode reward** — the total reward the agent collected in a finished
  episode. This is the headline "is it actually getting better?" signal: as the
  agent learns to fly the lander, this curve should trend upward over time.
* **Loss** — how far each learning update's prediction was from its target. This
  describes the *training dynamics* rather than performance: it tends to be noisy
  and does not have to go to zero, but watching it helps spot a run that has
  blown up (loss exploding) versus one that is settling.
* **Epsilon** — the current exploration rate. Plotting it confirms the agent is
  exploring a lot early and gradually shifting to exploiting what it has learned,
  exactly as the epsilon schedule intends.

There is also a single **evaluation mean reward** scalar, logged once after the
post-training evaluation pass, summarizing how the final greedy policy performed.

**Graceful degradation — why a logging failure must never stop training.**
Logging is a convenience, not the point of the program. If the log directory
can't be created (a bad path, a permissions problem), it would be absurd to throw
away a long training run just because we couldn't draw a chart. So this logger is
built to *fail soft*: if it cannot set up its writer, it prints an error naming
the directory, quietly switches into a disabled state, and turns every later
``log_*`` call into a harmless no-op. Training carries on exactly as before — just
without the pretty graphs. The constructor never raises for a directory problem.
"""

from __future__ import annotations

import os

from torch.utils.tensorboard import SummaryWriter

# Descriptive TensorBoard tags. The "section/name" convention groups related
# scalars together in the TensorBoard UI (an "episode" group, a "train" group,
# and an "eval" group), which keeps the dashboard readable.
_EPISODE_REWARD_TAG = "episode/reward"
_LOSS_TAG = "train/loss"
_EPSILON_TAG = "train/epsilon"
_EVAL_MEAN_REWARD_TAG = "eval/mean_reward"


class TensorBoardLogger:
    """A thin wrapper around PyTorch's ``SummaryWriter`` for DQN metrics.

    Each public method is named for the exact metric it records, so the calls in
    the training loop read like plain English (``logger.log_episode_reward(...)``,
    ``logger.log_loss(...)``). The caller supplies the step index for each metric,
    because the trainer is the one that knows how many episodes or updates have
    happened — this class just writes down whatever it is told.

    If the log directory cannot be created at construction time, the logger
    disables itself: it records that logging is off, prints an error naming the
    directory, and every subsequent ``log_*`` call does nothing. This is the
    graceful-degradation behavior described in the module docstring — logging
    problems never interrupt training.
    """

    def __init__(self, log_dir: str) -> None:
        """Create the log directory (and parents) and open the writer.

        We create the directory up front, including any missing parent folders,
        so the very first write has somewhere to go. If that creation fails for
        any reason, we do NOT raise: instead we print an error that names the
        directory, mark the logger as disabled, and leave the writer unset so
        every later log call becomes a no-op and training can continue.
        """

        # Remember which directory we were asked to use, purely so error messages
        # and any later debugging can name it.
        self.log_dir = log_dir

        # Start out assuming logging is on; we flip this to False if setup fails.
        self.logging_enabled = True

        # The underlying writer. Stays None while disabled so the no-op guard in
        # every log method has something clear to check.
        self.writer: SummaryWriter | None = None

        # The last epsilon value we actually wrote. Epsilon is only logged when
        # it changes, so we keep the previous value to compare against. None means
        # "we have not logged epsilon yet", which makes the first call always log.
        self._last_logged_epsilon: float | None = None

        try:
            # Create the directory including any missing parents. exist_ok=True
            # means "fine if it is already there", so re-running training does not
            # fail on an existing folder.
            os.makedirs(log_dir, exist_ok=True)

            # Open the TensorBoard writer pointed at that directory.
            self.writer = SummaryWriter(log_dir=log_dir)
        except OSError as error:
            # Directory creation (or writer setup) failed. Degrade gracefully:
            # surface a plain-language error naming the directory, then disable
            # logging so training is unaffected.
            self.logging_enabled = False
            self.writer = None
            print(
                f"[TensorBoardLogger] Could not create or open the log directory "
                f"'{log_dir}' ({error}). Logging is disabled for this run; "
                f"training will continue without TensorBoard metrics."
            )

    def log_episode_reward(self, episode_index: int, total_reward: float) -> None:
        """Record the total reward of a finished episode.

        ``episode_index`` is the per-episode counter the trainer maintains: it
        starts at 0 for the first completed episode and increments by one for
        each episode after that. ``total_reward`` is the sum of rewards the agent
        collected over that episode. This is the curve a learner watches to see
        whether the agent is improving over time.
        """

        # No-op if logging was disabled at construction.
        if not self.logging_enabled or self.writer is None:
            return

        self.writer.add_scalar(_EPISODE_REWARD_TAG, total_reward, episode_index)
        # Flush so the scalar is readable by TensorBoard promptly rather than
        # sitting in a buffer until the writer happens to flush on its own.
        self.writer.flush()

    def log_loss(self, update_step: int, loss: float) -> None:
        """Record the training loss for one completed Q-network update.

        ``update_step`` is the per-update counter: it starts at 0 for the first
        learning update and increments by one per update thereafter. ``loss`` is
        the value of the loss function for that update. Watching loss helps spot a
        run whose training has destabilized.
        """

        if not self.logging_enabled or self.writer is None:
            return

        self.writer.add_scalar(_LOSS_TAG, loss, update_step)
        self.writer.flush()

    def log_epsilon(self, update_step: int, epsilon: float) -> None:
        """Record the current exploration rate, but only when it has changed.

        Epsilon decays gradually, so logging it on every single update would
        write a flood of near-identical points. Instead we only record it when it
        differs from the value we last wrote. We track that previous value in
        ``self._last_logged_epsilon``; because it starts as ``None``, the very
        first call always logs. ``update_step`` tags the point so epsilon lines up
        with the loss curve on the same step axis.
        """

        if not self.logging_enabled or self.writer is None:
            return

        # Change detection: skip the write when epsilon is identical to the last
        # value we logged. The first call (last value is None) always passes.
        if self._last_logged_epsilon is not None and epsilon == self._last_logged_epsilon:
            return

        self.writer.add_scalar(_EPSILON_TAG, epsilon, update_step)
        self.writer.flush()

        # Remember what we just wrote so the next call can compare against it.
        self._last_logged_epsilon = epsilon

    def log_eval_mean_reward(self, mean_reward: float) -> None:
        """Record the evaluation pass's mean reward as a single scalar.

        The post-training evaluation runs the greedy policy for several episodes
        and averages their total rewards; this method writes that one summary
        number. It is logged at step 0 because there is only ever one such value
        per run — a single point, not a curve.
        """

        if not self.logging_enabled or self.writer is None:
            return

        self.writer.add_scalar(_EVAL_MEAN_REWARD_TAG, mean_reward, 0)
        self.writer.flush()

    def close(self) -> None:
        """Close the underlying writer, flushing anything still buffered.

        Safe to call even when logging was disabled: if there is no writer there
        is nothing to close, so this simply returns. The trainer calls this once
        at the end of a run so the event files are finalized cleanly.
        """

        if self.writer is not None:
            self.writer.close()
