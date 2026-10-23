"""The replay buffer and experience sampling — "what is experience replay?"

Why does this buffer exist at all? A DQN agent walks through the environment one
step at a time, and two steps taken back-to-back are almost always very similar:
the lander is in nearly the same place, moving in nearly the same way. If we
trained the Q-network on each step the moment it happened, we would feed it a
long run of highly *correlated* examples, and training on correlated data makes
learning unstable — the network keeps chasing whatever situation it just saw.

Experience replay fixes this with one simple idea: instead of learning from the
latest step, we *store* every transition the agent experiences and later draw a
**random** minibatch of past transitions to learn from. Random sampling mixes
together experiences from many different moments, which breaks the step-to-step
correlation. It also lets each experience be reused in many updates rather than
being seen once and thrown away, so the agent squeezes more learning out of the
same amount of interaction.

A transition is one unit of experience: ``(state, action, reward, next_state,
terminated)`` — "I was in ``state``, took ``action``, got ``reward``, landed in
``next_state``, and here is whether the episode ended there."

HOW it is stored: the buffer is a fixed-capacity **ring buffer** backed by
preallocated NumPy arrays. "Ring" refers to a write index that marches forward
and then *wraps around* to the beginning, overwriting the oldest entry once the
buffer is full. A tiny example with capacity 3:

    add A -> [A, _, _]   write index now 1, count 1
    add B -> [A, B, _]   write index now 2, count 2
    add C -> [A, B, C]   write index now 0 (wrapped!), count 3 (full)
    add D -> [D, B, C]   D overwrote A, the oldest entry; count stays 3

So once full, each new ``add`` evicts exactly the single oldest transition — that
wrap-around of the write index is the one non-obvious part of the mechanics, and
everything else is plain array assignment.

The buffer exposes two clearly separated jobs:

* :meth:`add` — "store one experience."
* :meth:`sample` — "draw a random learning batch."

:meth:`sample` returns ``None`` until the buffer has *warmed up* (holds at least
``batch_size`` transitions). That is deliberate: early in training there is not
yet enough decorrelated experience to form a full batch, so the trainer simply
skips learning and keeps collecting until the buffer is ready.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    """One minibatch of transitions drawn from the buffer, ready for learning.

    Each field is a NumPy array stacked along the first axis, so every field has
    length ``batch_size`` and the i-th entries across all fields describe the
    same single transition.

    * ``states``      — float32 array of shape ``(batch_size, obs_dim)``
    * ``actions``     — int64 array of shape ``(batch_size,)``
    * ``rewards``     — float32 array of shape ``(batch_size,)``
    * ``next_states`` — float32 array of shape ``(batch_size, obs_dim)``
    * ``terminateds`` — bool array of shape ``(batch_size,)``
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminateds: np.ndarray


class ReplayBuffer:
    """A fixed-capacity ring buffer of past transitions for experience replay.

    The buffer holds at most ``capacity`` transitions. New transitions are added
    with :meth:`add`; once the buffer is full, each further add overwrites the
    single oldest transition (first-in, first-out eviction). Learning batches are
    drawn with :meth:`sample`, which returns ``None`` until the buffer holds at
    least the requested ``batch_size`` transitions.

    The underlying storage is a set of preallocated NumPy arrays — one per
    transition field — sized to ``capacity``. The arrays for ``state`` and
    ``next_state`` are allocated lazily on the first :meth:`add`, because that is
    when the observation dimensionality first becomes known. Allocating them then
    means the constructor needs nothing more than the capacity, and the buffer
    adapts to whatever environment it is used with.
    """

    def __init__(self, capacity: int) -> None:
        # ``capacity`` is the fixed number of transitions the ring can hold. Past
        # this many, the oldest entry is overwritten on each new add.
        self.capacity = capacity

        # ``_size`` is how many transitions are currently stored (0..capacity).
        # ``_write_index`` is where the *next* transition will be written; it
        # marches forward and wraps back to 0 at ``capacity`` (the "ring").
        self._size = 0
        self._write_index = 0

        # The per-field storage arrays. They stay ``None`` until the first add,
        # at which point the observation dimension is known and they are
        # allocated. Reward/action/terminated arrays do not depend on obs_dim, so
        # they could be sized now, but we allocate them all together in one place
        # on the first add to keep the lazy-allocation story in a single spot.
        self._states: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._terminateds: np.ndarray | None = None

    def __len__(self) -> int:
        """Return how many transitions are currently stored (0 up to capacity)."""
        return self._size

    def _allocate(self, obs_dim: int) -> None:
        """Preallocate the storage arrays now that the observation size is known.

        Called once, from the first :meth:`add`. The state arrays are
        ``(capacity, obs_dim)`` because each state is a vector; the action,
        reward, and terminated arrays are one scalar per stored transition.
        """
        self._states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._terminateds = np.zeros(self.capacity, dtype=np.bool_)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition, overwriting the oldest if the buffer is full.

        The transition ``(state, action, reward, next_state, terminated)`` is
        written at the current write index. The write index then advances by one
        and wraps back to 0 when it reaches the capacity — that wrap is what makes
        a full buffer overwrite its oldest entry (FIFO eviction). While the buffer
        is below capacity the stored count grows by one per add; once full it
        stays pinned at ``capacity``.
        """
        # On the very first add we learn the observation dimensionality and
        # allocate the backing arrays to match it.
        if self._states is None:
            self._allocate(obs_dim=len(state))

        # Write the transition's fields into the current slot. NumPy copies the
        # values into the preallocated arrays, so the buffer does not hold onto
        # the caller's original objects.
        index = self._write_index
        self._states[index] = state
        self._actions[index] = action
        self._rewards[index] = reward
        self._next_states[index] = next_state
        self._terminateds[index] = terminated

        # Advance the write index, wrapping around to 0 at capacity so the next
        # add overwrites the oldest slot (the ring behavior described up top).
        self._write_index = (self._write_index + 1) % self.capacity

        # Grow the stored count until the buffer is full, then hold it steady at
        # capacity (each later add replaces an entry rather than adding one).
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch | None:
        """Draw a random minibatch of distinct transitions, or ``None`` if too few.

        If the buffer holds fewer than ``batch_size`` transitions it has not yet
        warmed up, so there is no full batch to return and this returns ``None``
        to tell the trainer to skip learning for now.

        Otherwise it selects exactly ``batch_size`` stored transitions uniformly
        at random **without replacement** (so every transition in the batch is
        distinct) and returns them stacked into a :class:`Batch`. The ``rng`` is a
        seeded :class:`numpy.random.Generator`, which keeps sampling reproducible
        across runs that use the same seed.
        """
        # Not warmed up yet: not enough experience for a full batch.
        if self._size < batch_size:
            return None

        # Choose ``batch_size`` distinct slot indices from the filled region
        # ``[0, self._size)``. ``replace=False`` guarantees no index repeats, so
        # the sampled transitions are all distinct.
        indices = rng.choice(self._size, size=batch_size, replace=False)

        # Gather the chosen transitions field-by-field into a Batch.
        return Batch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminateds=self._terminateds[indices],
        )
