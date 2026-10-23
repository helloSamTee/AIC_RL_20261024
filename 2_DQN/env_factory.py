"""Creating and seeding the LunarLander environment — where the agent "lives".

This module isolates *all* of the Gymnasium plumbing in one small place. Every
other file in this project — the networks, the replay buffer, the policy, the
training loop — works with an already-built environment and never has to think
about how it was created, which identifier it uses, or how video recording is
wired up. If you ever want to point the agent at a different environment, this is
the only file you need to touch.

Two factory functions live here, each named for *what it returns*:

* :func:`create_env` builds the plain ``LunarLander-v3`` environment the agent
  trains in.
* :func:`create_recording_env` builds the same environment wrapped so that
  episodes are saved to disk as ``.mp4`` video files — used by the evaluation
  pass so you can watch the trained agent fly.

LunarLander needs the Box2D physics backend, installed via ``gymnasium[box2d]``.
A fresh setup that forgot it is the most common way this file fails. We *do not*
catch or rewrap those setup errors: Gymnasium's own messages are already clear and
name the exact problem (a missing dependency, or an unregistered environment id),
so letting them surface untouched gives a learner the most useful hint. There is
no custom exception layer to see through.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


# The one place the environment identifier is written down. Reading it from a
# single named constant (rather than retyping the string) keeps both factory
# functions in agreement and makes it obvious where to change environments.
ENV_ID = "LunarLander-v3"


def create_env(seed: int | None, render: bool = False) -> gym.Env:
    """Build and seed the LunarLander environment the agent trains in.

    Steps, in order:

    1. Create the environment with ``gym.make(ENV_ID)``. If Box2D is missing,
       Gymnasium raises its own ``DependencyNotInstalled`` error naming the
       dependency; if ``LunarLander-v3`` is not registered in the installed
       Gymnasium, it raises its own "name not found" error. We let both surface
       untouched — those messages are the clearest possible explanation.
    2. ``reset(seed=seed)`` the environment. Seeding on reset is what makes a run
       reproducible: the same seed produces the same sequence of starting states.
       A reset that fails to return an observation simply lets its natural error
       surface rather than being second-guessed here.

    ``render=True`` requests human-visible rendering (a pop-up window) for
    interactive debugging. The evaluation pass does not use this — it records to
    disk via :func:`create_recording_env` instead.

    The caller reads the observation dimensionality and the action count off the
    returned environment (``env.observation_space.shape[0]`` and
    ``env.action_space.n``) rather than hard-coding 8 and 4, so the networks are
    sized from the environment itself.
    """

    # render_mode="human" opens a visible window; otherwise we don't render at
    # all during training (it would only slow the loop down).
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)

    # Seed the environment on reset so the starting state (and the environment's
    # own randomness) are reproducible for a given seed. If seed is None,
    # Gymnasium picks a fresh random seed, giving a non-reproducible run.
    env.reset(seed=seed)

    return env


def create_recording_env(
    video_dir: str,
    seed: int | None,
    name_prefix: str = "eval",
    episode_trigger=lambda episode_index: True,
) -> gym.Env:
    """Build a LunarLander environment that saves episodes to disk as video.

    This is the environment the evaluation pass runs in, so we can watch how the
    trained agent behaves. It is the plain environment plus one wrapper:

    1. The environment is created with ``render_mode="rgb_array"`` — that tells
       Gymnasium to render each frame as an image array (rather than to a screen)
       so the frames can be collected into a video file.
    2. It is wrapped in Gymnasium's built-in ``RecordVideo`` wrapper, which
       watches for episodes to finish and writes each recorded episode to an
       ``.mp4`` file in ``video_dir``. The ``name_prefix`` becomes the start of
       each video's filename, and ``episode_trigger`` decides *which* episodes to
       record: it receives the episode index and returns True to record that
       episode. The default records every episode.

    ``RecordVideo`` creates ``video_dir`` (including any missing parent
    directories) if it does not exist, so the caller does not need to make the
    folder first.

    As in :func:`create_env`, we seed on reset for reproducibility and let any
    Gymnasium setup error surface untouched.
    """

    # "rgb_array" renders frames as image arrays off-screen, which is what the
    # recorder needs to stitch together into a video.
    base_env = gym.make(ENV_ID, render_mode="rgb_array")

    # Wrap the environment so finished episodes are written out as .mp4 files.
    # video_folder is created (with parents) by the wrapper if it's missing.
    recording_env = RecordVideo(
        base_env,
        video_folder=video_dir,
        name_prefix=name_prefix,
        episode_trigger=episode_trigger,
    )

    # Seed on reset for a reproducible evaluation run, same as the training env.
    recording_env.reset(seed=seed)

    return recording_env
