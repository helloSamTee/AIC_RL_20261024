"""The run — read this file LAST.

Every other module in this package taught one DQN idea on its own: the
:class:`~config.Config` of knobs, the :class:`~networks.QNetwork` brain and its
frozen target-network twin, the :class:`~replay_buffer.ReplayBuffer` of past
experience, the epsilon-greedy :mod:`policy`, the :func:`~td_target.compute_td_target`
math, the checkpoint snapshot, and the :class:`~logger.TensorBoardLogger`. The
:class:`~trainer.Trainer` then wired them into the online training loop, and
:func:`~evaluate.evaluate` scored the finished policy with exploration turned off.

This file is where the whole program finally runs, top to bottom, so a learner can
trace one complete run from a single place:

    build the Config  ->  validate it  ->  construct the Trainer
      ->  train  ->  evaluate  ->  close the logger

There is **no command-line interface by design** (Requirement 10.6). You do not pass
hyperparameters as flags; you edit them in code. The one place to change how the agent
trains is the ``Config`` built in :func:`main` below — every knob there falls back to a
documented default from :mod:`config`, so you only write down the ones you want to
change. Keeping configuration in code (rather than argparse) means the exact settings a
run used are visible right here in the source, not buried in a shell command.

The construction is deliberately explicit and in reading order rather than hidden behind
factory indirection, so this file reads like a narration of the run itself.
"""

from __future__ import annotations

from config import Config
from evaluate import evaluate
from trainer import Trainer


def main() -> None:
    """Run one complete DQN experiment: build config, validate, train, evaluate, close.

    The body below is the whole program in order. Each step is a single, clearly
    named call with a comment explaining *why* it is here, so the run reads as a
    short story a learner can follow line by line.
    """

    # --- Step 1: build the configuration, in code ------------------------
    # This is the single place to edit hyperparameters. Construct Config with any
    # overrides you want as plain literals, e.g.:
    #
    #     cfg = Config(total_steps=200_000, seed=0, learning_rate=5e-4)
    #
    # Any field you leave out keeps its documented default from config.py. We pass no
    # overrides here, so this run uses the defaults exactly as written in that file.
    cfg = Config()

    # --- Step 2: validate the configuration, out loud --------------------
    # Validate before building anything so a bad knob fails immediately, with a clear
    # message naming the offending field, instead of crashing deep inside training.
    # The Trainer constructor re-validates internally; we also call it here as its own
    # visible step so a learner sees configuration validation as an explicit beat of
    # the program rather than a hidden side effect. The duplicate call is harmless.
    cfg.validate()

    # --- Step 3: construct the Trainer -----------------------------------
    # Constructing the Trainer builds everything the loop needs, wired together in one
    # place: it seeds the RNGs, creates the LunarLander environment, the online
    # Q-network and its frozen target twin, the optimizer, the replay buffer, and the
    # TensorBoard logger (which the Trainer owns as `trainer.logger`). After this line
    # the agent is fully assembled and ready to learn.
    trainer = Trainer(cfg)

    # Steps 4-6 are wrapped in try/finally so the logger is ALWAYS closed — even if
    # training or evaluation raises partway through. Closing flushes the TensorBoard
    # event files to disk; skipping it on an error could lose the metrics of a long run.
    try:
        # --- Step 4: train ----------------------------------------------
        # Run the online interaction-and-learning loop for cfg.total_steps steps. When
        # it finishes, train() saves a checkpoint of the trained Q-network to
        # cfg.checkpoint_path (the evaluation pass reloads it from there).
        trainer.train()

        # --- Step 5: evaluate -------------------------------------------
        # Measure the trained policy with exploration turned OFF (greedy actions),
        # recording at least one episode as video. We reuse the Trainer's own logger so
        # the evaluation score lands in the same TensorBoard run as the training curves.
        # evaluate() returns the mean total reward across the evaluation episodes; we
        # print it so the headline result is visible in the console too.
        mean_reward = evaluate(cfg, trainer.logger)
        print(f"Evaluation mean reward over {cfg.eval_episodes} episodes: {mean_reward:.2f}")
    finally:
        # --- Step 6: close the logger -----------------------------------
        # Always flush and close the TensorBoard writer so the event files are
        # finalized. This is safe to call even if logging was disabled at startup.
        trainer.logger.close()


# --- Step 7: the in-code entry point -------------------------------------
# Running `python main.py` from inside the 2_DQN directory starts the whole run via
# main(). There is intentionally no argument parsing here — configuration lives in the
# Config built above, not on the command line (Requirement 10.6).
if __name__ == "__main__":
    main()
