"""Saving and restoring the Q-network's weights — "what is a checkpoint?"

A **checkpoint** is a saved snapshot of the Q-network's learned weights, written
to a file on disk. Training the agent can take a long time, so once it has
learned something useful we don't want to throw that away when the program ends.
A checkpoint lets us persist the trained policy and reuse it later — for example,
the evaluation pass reloads these weights to watch the agent play — without
having to retrain from scratch.

What actually gets saved is the network's ``state_dict``: PyTorch's plain
dictionary mapping each parameter's name (like ``"net.0.weight"``) to its current
tensor of numbers. Saving the ``state_dict`` rather than the whole Python object
is the recommended, portable way to persist a model's learned values.

Why bother making the save **atomic**? If we wrote the weights straight into the
final file and the program crashed (or the disk filled up) halfway through, we'd
be left with a half-written, corrupt checkpoint — and we'd only discover that
later, when a load fails. To avoid that, we write to a *temporary* file first and
then rename it into place. A rename within the same directory is atomic on the
operating system: the destination file either still holds the old contents or
holds the complete new contents, never a half-written mix. So a crash can leave a
stray temp file lying around, but it can never leave a broken model at the real
path.

Failures here surface as Python's own built-in errors (``OSError`` when a write
fails, ``FileNotFoundError`` when a load target is missing) so the message names
the offending path directly, instead of hiding behind a custom exception.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def save_checkpoint(q_network: nn.Module, path: str) -> None:
    """Save the Q-network's weights to ``path`` using an atomic write.

    The steps, in order:

    1. Make sure the directory that will hold the checkpoint exists, creating any
       missing parent directories along the way. This covers the common case
       where the configured path points at a ``checkpoints/`` folder that has not
       been created yet.
    2. Write the network's ``state_dict`` to a *temporary* file that lives in the
       SAME directory as the final path. Being in the same directory is what lets
       the final rename be atomic.
    3. Atomically move the temp file onto the final path with ``os.replace``. If a
       file already exists at ``path`` it is replaced; either way the real path
       only ever holds a complete checkpoint.

    If writing fails (for example, the disk is full or the path is not writable),
    the natural ``OSError`` is allowed to propagate so its message — which names
    the file path — reaches the caller. We do not wrap it in a custom exception.
    """

    # Step 1: ensure the parent directory exists.
    # os.path.dirname("checkpoints/model.pt") -> "checkpoints".
    # If the path has no directory part (a bare filename), dirname is "" and we
    # simply save into the current working directory.
    parent_directory = os.path.dirname(path)
    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)

    # Step 2: write to a temporary file in the SAME directory as the final path.
    # Keeping the temp file beside the destination guarantees the rename in step 3
    # stays within one filesystem, which is what makes it atomic.
    temp_path = path + ".tmp"
    torch.save(q_network.state_dict(), temp_path)

    # Step 3: atomically move the completed temp file into place.
    os.replace(temp_path, path)


def load_checkpoint(q_network: nn.Module, path: str) -> None:
    """Restore the Q-network's weights from the checkpoint file at ``path``.

    This reads the ``state_dict`` previously written by :func:`save_checkpoint`
    and copies those saved values back into ``q_network`` in place. After this
    returns, every parameter in ``q_network`` equals the value that was saved, so
    the network behaves exactly as it did at save time.

    If ``path`` does not exist, the natural ``FileNotFoundError`` is allowed to
    surface — its message names the missing path, which is exactly the hint a
    learner needs when they point evaluation at a checkpoint that was never
    created.
    """

    # Read the saved parameter dictionary back from disk. If the file is missing,
    # torch.load raises FileNotFoundError naming the path, which we let propagate.
    saved_state_dict = torch.load(path)

    # Copy the saved values into the given network's parameters in place.
    q_network.load_state_dict(saved_state_dict)
