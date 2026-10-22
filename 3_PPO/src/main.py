"""The run — "how do all the pieces come together to train and watch an agent?"

This is the file a learner reads *last*. Everything before it built one PPO concept
per module — the config, the shared CNN encoder with its Actor and Critic heads, the
Gaussian tanh-squashed policy, the single-use rollout buffer with GAE, and the clipped
PPO update. ``main.py`` is where those pieces are wired into an actual run: pick a
device, build the environment and networks, train with the collect → update → clear
loop, log to TensorBoard, save a checkpoint, and finally watch the trained policy drive
in a deterministic evaluation pass with recorded video.

The module is assembled in small, well-named steps so the whole program can be traced
from one place. This first section covers **device selection**, which is what lets the
exact same code run on your laptop and on the hosted ``spark.enverge.ai`` GPU box
without edits.

Why device selection matters for a workshop
--------------------------------------------
Deep RL leans on tensor math that a GPU accelerates enormously, but not every learner
has a CUDA GPU in front of them. Rather than ask you to edit the code depending on
where it runs, the agent *chooses the device for you*: it uses CUDA when a CUDA-capable
GPU is reported available, and otherwise falls back to the CPU. Same files, same
commands, both places — the only difference is speed.

Why the CUDA-to-CPU fallback never aborts startup
--------------------------------------------------
Occasionally a machine *reports* a CUDA GPU but cannot actually initialize work on it
(a mismatched driver, a GPU that is busy or out of memory, a flaky remote box). If we
trusted ``torch.cuda.is_available()`` blindly and it then failed on first use, the whole
run would crash before training ever began. To make startup robust, :func:`select_device`
does a tiny *probe* on CUDA inside a ``try``/``except``: if the probe fails, it quietly
falls back to the CPU and lets startup continue instead of terminating (Req 12.2). A
learner on a half-broken GPU box still gets a (slower) training run rather than a
traceback.

Once the device is chosen, *every* tensor the agent creates — network inputs, network
outputs, and the network parameters themselves — must live on that one device, because
an operation whose operands sit on different devices fails (Req 12.3, 12.4). The small
:func:`move_networks_to_device` helper gives later sections a single, obvious place that
performs that placement for the encoder, Actor, and Critic.
"""

from __future__ import annotations

import math
import os
import random
from collections import namedtuple
from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

from config import Config
from networks import ActorHead, CNNEncoder, CriticHead, scale_pixels
from policy import clamp_to_bounds, deterministic_action, sample_action, tanh_rescale
from ppo import PPOUpdater
from rollout_buffer import RolloutBuffer, normalize_advantages


def select_device() -> torch.device:
    """Pick the compute device: CUDA when it genuinely works, otherwise CPU.

    The selection has two stages, and the second is what makes it robust:

    1. Ask PyTorch whether a CUDA-capable GPU is reported available
       (``torch.cuda.is_available()``). If it is not, there is nothing to probe —
       we simply return the CPU device (Req 12.1).
    2. If CUDA *is* reported available, don't trust that report blindly. Run a tiny
       **probe**: allocate a one-element tensor on CUDA inside a ``try``/``except``.
       If that probe succeeds, CUDA really works, so we return the CUDA device. If
       the probe raises for any reason (bad driver, busy or out-of-memory GPU, a
       flaky remote machine), we swallow the error, fall back to the CPU, and let
       startup continue instead of crashing (Req 12.2).

    Returns the selected :class:`torch.device`. The caller logs its name once so the
    learner can confirm which device the run actually used.
    """

    # Stage 1: no CUDA reported -> the answer is simply CPU.
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Stage 2: CUDA is reported available, so verify it with a tiny probe before we
    # commit to it. A single allocation is enough to trigger CUDA initialization; if
    # that initialization is going to fail, we want to find out here, safely, rather
    # than mid-training.
    try:
        _probe = torch.zeros(1, device="cuda")
        # Touch the result so the allocation is not optimized away and any lazy CUDA
        # initialization is actually forced to run.
        del _probe
        return torch.device("cuda")
    except Exception:
        # CUDA was reported but could not be initialized. Fall back to CPU and keep
        # going — a slower run is far better than aborting startup (Req 12.2).
        return torch.device("cpu")


def move_networks_to_device(
    device: torch.device, *modules: nn.Module
) -> tuple[nn.Module, ...]:
    """Place one or more networks on the selected device and hand them back.

    PyTorch's ``module.to(device)`` moves every parameter (and buffer) of a network
    onto the given device. Once the networks live on the device, so do their outputs,
    and we feed them device-resident inputs — which keeps every operand of every
    operation on the *same* device, the condition PyTorch requires (Req 12.3, 12.4).

    Collecting that placement into one tiny helper means the rest of ``main.py`` has a
    single, obvious line that does device placement for the encoder, Actor, and Critic
    together, e.g. ``encoder, actor, critic = move_networks_to_device(device, encoder,
    actor, critic)``. The modules are returned in the same order they were passed.
    """

    moved = tuple(module.to(device) for module in modules)
    return moved


# ---------------------------------------------------------------------------
# TensorBoard logging
# ---------------------------------------------------------------------------
#
# Why logging lives behind its own small class
# ---------------------------------------------
# TensorBoard is how a learner *watches* training happen: episode reward climbing,
# the policy/value losses settling, the entropy bonus shrinking as the policy grows
# confident. But logging is **observability, not training**. If writing a metric ever
# fails, the right thing to do is keep training and lose a little visibility — never to
# crash a run that may have taken many minutes to reach that point.
#
# That principle is called *graceful degradation*, and this class is built around it:
#
#   * Every method returns an **error indication** instead of raising: it returns
#     ``None`` on success, or a short human-readable error string on failure. The
#     string names exactly what went wrong (which directory, or which tag + index),
#     so the failure is still a teaching moment even though it does not stop the run.
#     Each returned error string is also printed as a warning so it is visible in the
#     console without the caller having to inspect the return value.
#   * If the log directory cannot be created at construction time, the logger enters a
#     permanently **disabled** state (``enabled = False``). Every log method then does
#     nothing and returns immediately, so a broken log location silently costs you
#     metrics but never costs you the training run (Req 8.5).
#   * Before writing any numeric scalar, a non-finite guard checks ``math.isfinite``.
#     A NaN or infinite value is skipped (writing it would poison the TensorBoard plot)
#     with an error indication naming the affected tag and index, and training continues
#     (Req 8.6).
#
# Each public method is named for the exact metric it records, so the training loop
# reads as a list of plainly-labelled observations rather than a pile of generic
# ``add_scalar`` calls.


class TensorBoardLogger:
    """Record PPO training and evaluation metrics to TensorBoard, degrading gracefully.

    Construction creates the log directory (including any missing parents) and opens a
    :class:`torch.utils.tensorboard.SummaryWriter` pointed at it (Req 8.4). If the
    directory cannot be created, the logger does **not** raise — it enters a disabled
    state in which every log method is a no-op, so an unusable log location never stops
    training (Req 8.5).

    Error-indication convention (consistent across every method):
        * return ``None`` on success,
        * return a short error string on failure (and print it as a warning).

    The logger never raises for a directory-creation failure or a non-finite value;
    graceful degradation is the whole point.
    """

    # Stable, human-readable tags. Keeping them as attributes means every metric has
    # exactly one tag spelled in exactly one place.
    _TAG_EPISODE_REWARD = "train/episode_reward"
    _TAG_POLICY_LOSS = "train/policy_loss"
    _TAG_VALUE_LOSS = "train/value_loss"
    _TAG_ENTROPY = "train/entropy"
    _TAG_DEVICE = "run/device"
    _TAG_EVAL_REWARD = "eval/mean_reward"

    def __init__(self, log_dir: str) -> None:
        """Open a writer on ``log_dir``, creating the directory tree first (Req 8.4).

        On a directory-creation failure the logger is left disabled rather than raising
        (Req 8.5): ``self.enabled`` is ``False``, ``self.writer`` is ``None``, and the
        error that would have been reported is stored on ``self.init_error`` (and
        printed) so the caller can see which directory could not be created.
        """

        self.log_dir = log_dir
        # Tracks whether log_device has already recorded the device name, so repeated
        # calls record it exactly once (Req 12.5).
        self._device_logged = False
        # Populated with the error string if construction fails to create the directory.
        self.init_error: str | None = None

        # Create the directory tree *before* constructing the writer (Req 8.4). If this
        # fails we disable the logger instead of letting the OSError escape (Req 8.5).
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as error:
            self.enabled = False
            self.writer = None
            self.init_error = (
                f"TensorBoard logging disabled: could not create log directory "
                f"'{log_dir}' ({error}). Training will continue without metrics."
            )
            print(self.init_error)
            return

        # Directory is ready; open the writer and mark the logger live.
        self.writer = SummaryWriter(log_dir=log_dir)
        self.enabled = True

    def _write_scalar(self, tag: str, value: float, index: int) -> str | None:
        """Write one numeric scalar, guarding against a disabled logger and non-finite values.

        This is the single choke point every scalar-logging method flows through, so the
        two cross-cutting rules live in one place:

        * If the logger is disabled, do nothing and return ``None`` (a disabled logger is
          not an error for the caller — it simply records nothing).
        * If ``value`` is not finite (NaN or infinite), skip the write and return an error
          string naming the tag and index, allowing training to continue (Req 8.6).
          Writing a non-finite value would otherwise corrupt the TensorBoard plot.
        """

        if not self.enabled:
            return None

        if not math.isfinite(value):
            error = (
                f"Skipped non-finite scalar for tag '{tag}' at index {index} "
                f"(value={value}); training continues."
            )
            print(error)
            return error

        self.writer.add_scalar(tag, value, index)
        return None

    def log_episode_reward(self, episode_index: int, total_reward: float) -> str | None:
        """Record one episode's total reward, tagged by the episode index (Req 8.1).

        The *caller* owns the episode index: it starts at 0 and increments by 1 for each
        completed episode. Returns ``None`` on success, or an error string if the reward
        is non-finite and was skipped.
        """

        return self._write_scalar(self._TAG_EPISODE_REWARD, total_reward, episode_index)

    def log_update(
        self,
        update_phase_index: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
    ) -> str | None:
        """Record the three update-phase metrics as separate scalars (Req 8.2).

        Policy loss, value loss, and entropy are each written under their own tag, all
        tagged by ``update_phase_index`` (the caller starts it at 0 and increments by 1
        per completed update phase). Each scalar is guarded independently, so one
        non-finite metric does not prevent the others from being written.

        Returns ``None`` when all three are written, or a combined error string naming
        each tag/index that was skipped for being non-finite.
        """

        errors = [
            self._write_scalar(self._TAG_POLICY_LOSS, policy_loss, update_phase_index),
            self._write_scalar(self._TAG_VALUE_LOSS, value_loss, update_phase_index),
            self._write_scalar(self._TAG_ENTROPY, entropy, update_phase_index),
        ]
        reported = [error for error in errors if error is not None]
        if reported:
            return " ".join(reported)
        return None

    def log_device(self, device_name: str) -> str | None:
        """Record the selected device name exactly once (Req 12.5).

        An internal already-logged flag guards against re-logging: the first call writes
        the name, and every later call is a no-op. Because a device name is a *string*
        and TensorBoard scalars are numeric, this is written with ``add_text`` (a text
        summary), which is the natural way to record a textual run attribute.

        Returns ``None`` on success or when the logger is disabled / already logged.
        """

        if not self.enabled:
            return None
        if self._device_logged:
            # Already recorded once — do not re-log (Req 12.5).
            return None

        self.writer.add_text(self._TAG_DEVICE, device_name)
        self._device_logged = True
        return None

    def log_eval_reward(self, mean_reward: float) -> str | None:
        """Record the mean evaluation reward as a single scalar (Req 11.6).

        Logged once per evaluation pass at index 0 (there is a single evaluation result).
        Returns ``None`` on success, or an error string if the value is non-finite and
        was skipped.
        """

        return self._write_scalar(self._TAG_EVAL_REWARD, mean_reward, 0)

    def close(self) -> None:
        """Flush buffered scalars and close the writer if the logger is enabled.

        Flushing matters because TensorBoard buffers writes; without a flush the most
        recent metrics of a short run might never reach disk. A disabled logger has no
        writer to close, so this is a safe no-op in that case.
        """

        if self.enabled and self.writer is not None:
            self.writer.flush()
            self.writer.close()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
#
# What is a checkpoint, in plain terms?
# -------------------------------------
# A *checkpoint* is a saved snapshot of the three networks' learned weights — the
# CNN encoder's "eyes", the Actor's policy, and the Critic's value estimator —
# written to a single file on disk. Training a PPO agent on CarRacing can take a long
# time, so once it is done we persist what it learned. With a checkpoint you can reuse
# a trained policy later (e.g. to evaluate it or watch it drive) *without retraining
# from scratch* (Req 10.1).
#
# Why all three networks must be present to be valid
# ---------------------------------------------------
# The agent is only whole when all three networks are restored together. The Actor
# decides how to act, the Critic judges states, and both read features produced by the
# shared encoder. If a checkpoint were missing any one of the three, loading it would
# leave that network sitting at its *random initial weights* — the agent would look
# "loaded" but silently behave like an untrained (or half-trained) one. That failure is
# quiet and very confusing. So :func:`load_checkpoint` treats any file lacking all three
# networks as invalid, refuses it, and leaves the in-memory networks untouched rather
# than partially overwriting them (Req 10.6).


# The three keys every valid checkpoint must contain, one per network. Kept as a module
# constant so the save side and the load-side validation agree on exactly one spelling.
_CHECKPOINT_KEYS = ("encoder", "actor", "critic")


def save_checkpoint(
    path: str, encoder: nn.Module, actor: nn.Module, critic: nn.Module
) -> None:
    """Save a snapshot of all three networks' weights to a single file at ``path``.

    The snapshot is one file holding the ``state_dict`` of each network under its own
    key — ``"encoder"``, ``"actor"``, ``"critic"`` — so a later
    :func:`load_checkpoint` can restore every network from one place (Req 10.1).

    Steps:

    1. **Ensure the destination directory exists.** If ``path`` names a directory that
       does not yet exist, create it along with any missing parents (Req 10.2). When
       ``path`` has no directory component (just a bare filename in the current
       directory), there is nothing to create, so that case is skipped.
    2. **Write the three state dicts in one file** via ``torch.save``.

    On a write failure, the raised error (which names ``path``) is allowed to propagate;
    we do not catch it. ``torch.save`` writes to the target path directly, so a failing
    write naturally surfaces the error and leaves any previously saved checkpoint at that
    path unchanged (Req 10.3).
    """

    # Step 1: make sure the parent directory exists before writing (Req 10.2). A bare
    # filename has an empty dirname; guard against that so we do not call makedirs("").
    parent_directory = os.path.dirname(path)
    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)

    # Step 2: collect each network's parameters under its own key and write the single
    # checkpoint file. Any write error propagates (it names the path) — Req 10.3.
    checkpoint = {
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str, encoder: nn.Module, actor: nn.Module, critic: nn.Module
) -> None:
    """Restore all three networks' weights from the checkpoint file at ``path``.

    This is the mirror of :func:`save_checkpoint`. It restores the encoder, Actor, and
    Critic so their parameters are identical to the ones that were saved (Req 10.4).

    Failure handling is deliberately careful so a bad load never leaves the agent in a
    half-restored state:

    1. **Missing file.** If ``path`` does not exist, a ``FileNotFoundError`` naming the
       missing path is allowed to propagate (Req 10.5).
    2. **Unreadable or incomplete file.** If the file cannot be read, or does not contain
       parameters for *all three* networks (keys ``"encoder"``, ``"actor"``, ``"critic"``),
       a ``ValueError`` is raised reporting that the checkpoint is invalid. Crucially, this
       validation happens *before* any ``load_state_dict`` call, so an invalid or partial
       checkpoint never mutates the in-memory networks — they are left exactly as they were
       (Req 10.6).
    3. **Valid file.** Only once all three keys are confirmed present do we restore every
       network, so the three loads happen together or not at all (Req 10.4).
    """

    # Step 1: a missing path is a file-not-found condition. Raise it explicitly so the
    # error clearly names the missing path (Req 10.5).
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Step 2: read the file. If it cannot be parsed as a checkpoint, treat it as an
    # invalid checkpoint — do NOT touch the in-memory networks (Req 10.6).
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as error:
        raise ValueError(
            f"Invalid checkpoint at '{path}': the file could not be read ({error})."
        ) from error

    # Step 3: validate the shape of the loaded object BEFORE restoring anything. It must
    # be a mapping that contains all three network keys. Checking first guarantees a
    # partial/invalid checkpoint never mutates any network's parameters (Req 10.6).
    missing_keys = [
        key
        for key in _CHECKPOINT_KEYS
        if not isinstance(checkpoint, dict) or key not in checkpoint
    ]
    if missing_keys:
        raise ValueError(
            f"Invalid checkpoint at '{path}': missing parameters for "
            f"{missing_keys}. A valid checkpoint must contain all three networks "
            f"{list(_CHECKPOINT_KEYS)}; the in-memory networks were left unchanged."
        )

    # Step 4: all three are present — restore them together so the agent is whole and
    # its parameters match the saved ones identically (Req 10.4).
    encoder.load_state_dict(checkpoint["encoder"])
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])


# ---------------------------------------------------------------------------
# Environment wiring
# ---------------------------------------------------------------------------
#
# Why the environment is built behind a tiny factory
# ---------------------------------------------------
# Everything the agent learns from flows out of the Gymnasium ``CarRacing-v3``
# environment. :func:`make_env` is the single place that constructs it, so the
# training loop and the (later) evaluation pass both ask for an environment the same
# way. CarRacing is configured for *continuous* actions (three real numbers:
# steering, gas, brake) and 96x96x3 RGB image observations (Req 1.1).
#
# Why this function deliberately catches *nothing*
# -------------------------------------------------
# A learner whose setup is incomplete should get the clearest possible signal. So
# three very different failures are all left to propagate untouched:
#
#   * a missing dependency (Box2D isn't installed) raises an ImportError /
#     DependencyNotInstalled that names the missing package (Req 1.4),
#   * an unregistered id (a typo, or a gymnasium too old to know ``CarRacing-v3``)
#     raises an error saying the id is unknown (Req 1.5),
#   * a reset that fails to yield a valid initial observation surfaces as its own
#     error (Req 1.7).
#
# Catching any of these would only turn a precise, actionable message into a vaguer
# one. The factory's whole job is to construct and hand back — never to hide a setup
# problem.


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create the ``CarRacing-v3`` environment configured for continuous control.

    ``render_mode`` is passed straight through to Gymnasium: ``None`` for the
    (fast, windowless) training environment, or ``"rgb_array"`` for the evaluation
    pass that records video without opening a screen.

    No error is caught here on purpose (Req 1.4, 1.5): a missing Box2D dependency,
    an unregistered environment id, and any construction error all propagate with
    their original, descriptive messages so a learner can fix their setup directly.
    """

    # continuous=True selects CarRacing's 3-D continuous action space. Any failure
    # (missing dependency, unknown id) is left to propagate — see the section note.
    return gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)


# A small, named bundle of the facts we read from the environment once at startup:
# the observation shape, the action dimensionality, and the per-dimension action
# bounds. Naming them (rather than returning a bare tuple) keeps every later use —
# building the buffer, rescaling actions — readable and self-documenting.
EnvSpec = namedtuple("EnvSpec", ["observation_shape", "action_dim", "action_low", "action_high"])


def read_env_spec(env: gym.Env) -> EnvSpec:
    """Read and retain the observation shape, action dimensionality, and action bounds.

    These facts are read *from the environment* rather than hard-coded (Req 1.2), and
    retained so later steps can validate observation shapes and rescale/clamp actions
    to the real bounds (Req 1.3). For CarRacing this yields an observation shape of
    ``(96, 96, 3)``, an action dimensionality of 3, and per-dimension low/high bounds.

    Returns an :class:`EnvSpec` of ``(observation_shape, action_dim, action_low,
    action_high)`` where the two bounds are NumPy arrays of length ``action_dim``.
    """

    # Observation geometry straight from the env's observation space.
    observation_shape = tuple(env.observation_space.shape)

    # The continuous action space tells us how many action components there are and
    # the inclusive [low, high] bound for each one.
    action_dim = env.action_space.shape[0]
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    return EnvSpec(observation_shape, action_dim, action_low, action_high)


# ---------------------------------------------------------------------------
# Seeding (reproducibility)
# ---------------------------------------------------------------------------
#
# Why seeding lives in one helper
# --------------------------------
# A reproducible run means: same seed + same hyperparameters + same device ->
# identical action sequence and identical learned parameters (Req 13.2). Randomness
# in this project enters through four doors, so all four must be seeded together or
# reproducibility silently breaks:
#
#   1. Python's ``random`` module,
#   2. NumPy's global RNG,
#   3. torch (network initialization happens through torch's global RNG),
#   4. the environment (its reset/step stochasticity, seeded via ``env.reset(seed=)``).
#
# On top of those globals we build and return TWO explicit generators, because the
# two places that must be *exactly* reproducible take an explicit generator rather
# than leaning on a global:
#   * a ``torch.Generator`` for action sampling (policy.sample_action), and
#   * a ``numpy.random.Generator`` for minibatch shuffling inside the PPO update.
#
# Deriving both from the same seed is what ties the whole run to that one number.


# A named bundle for the two explicit generators the run threads through to the two
# reproducibility-critical spots (action sampling and minibatch shuffling).
Generators = namedtuple("Generators", ["torch_generator", "numpy_generator"])


def seed_everything(seed: int | None, env: gym.Env, device: torch.device) -> Generators:
    """Seed every source of randomness and return the two explicit generators.

    When ``seed`` is an integer, all four randomness sources (Python ``random``,
    NumPy global, torch global, and the environment) are seeded from it, and the two
    returned generators are seeded from it too — so two runs with the same seed on the
    same device produce identical action sequences and identical network parameters
    (Req 13.2). When ``seed`` is ``None`` we still build both generators (unseeded /
    default), so the training loop always has concrete generators to pass along.

    The torch generator is created **on ``device``**. ``torch.randn(..., generator=g,
    device=d)`` requires the generator and the target tensor to live on the same
    device, so when the run is on CUDA the sampling generator must be a CUDA generator
    too — otherwise action sampling raises
    ``RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'``.

    Returns a :class:`Generators` of ``(torch_generator, numpy_generator)``.
    """

    # Seed the environment's own stochasticity. Gymnasium takes the seed on reset;
    # passing seed=None is valid and simply leaves the env unseeded.
    env.reset(seed=seed)

    if seed is not None:
        # Seed the three global RNGs so anything relying on them (notably torch's
        # network initialization) is reproducible.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Explicit generators for the two reproducibility-critical consumers. Seeding
        # them from the same seed ties action sampling and minibatch shuffling to it.
        # The torch generator lives on the run device so it matches the on-device
        # tensors it draws noise for in sample_action.
        torch_generator = torch.Generator(device=device)
        torch_generator.manual_seed(seed)
        numpy_generator = np.random.default_rng(seed)
    else:
        # No seed: still hand back concrete, default (unseeded) generators so the
        # loop's call sites are identical whether or not a seed was supplied.
        torch_generator = torch.Generator(device=device)
        numpy_generator = np.random.default_rng()

    return Generators(torch_generator, numpy_generator)


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------
#
# The on-policy rhythm, in one sentence
# -------------------------------------
# PPO's update math is only valid for data gathered by the *current* policy, so every
# iteration must: COLLECT exactly one horizon of fresh experience, turn it into
# learning targets with GAE, run ONE update phase on it, and then CLEAR the buffer so
# none of that now-stale data leaks into the next iteration. Collect -> GAE -> update
# -> clear, repeat. The helpers below are named for those four beats so the loop body
# reads like that sentence.


# What :func:`train` hands back to the entry point / evaluation (task 8.6 wires this
# into evaluate()). It deliberately does NOT include or close the logger-owning
# lifecycle beyond returning the logger; the entry point closes the logger after
# evaluation.
TrainResult = namedtuple(
    "TrainResult",
    ["encoder", "actor", "critic", "action_low", "action_high", "logger", "device"],
)


def _observation_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Turn one raw ``(96, 96, 3)`` uint8 observation into a network-ready tensor.

    The encoder wants a *scaled*, *channels-first*, *batched* float tensor on the
    compute device: shape ``(1, 3, 96, 96)`` with pixels in 0.0..1.0. We add the batch
    dimension, scale the pixels (Req 2.6), and permute H,W,C -> C,H,W.
    """

    # (96,96,3) uint8 -> (1,96,96,3) uint8 tensor on the device.
    obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
    # Scale 0..255 -> 0.0..1.0, then move channels to the front: (1,3,96,96).
    scaled = scale_pixels(obs_tensor)
    return scaled.permute(0, 3, 1, 2).contiguous()


def _select_action(
    obs: np.ndarray,
    encoder: CNNEncoder,
    actor: ActorHead,
    critic: CriticHead,
    low: torch.Tensor,
    high: torch.Tensor,
    torch_generator: torch.Generator,
    device: torch.device,
):
    """Choose one exploratory action for ``obs`` and report everything the buffer needs.

    This is the per-step action-selection pipeline, isolated so the collection loop
    reads clearly. For the current observation it runs the shared encoder once, asks
    the Actor for a Gaussian ``(mean, std)`` and the Critic for a value, then *samples*
    (training wants exploration) a raw pre-squash action and its finite log-prob. The
    raw action is squashed+rescaled onto the real bounds and clamped so it is always a
    legal environment action (Req 3.1, 3.2, 3.3, 3.5).

    Returns ``(env_action, raw_action, log_prob, value)`` where ``env_action`` is a
    NumPy vector ready for ``env.step`` and ``raw_action`` is the pre-squash vector the
    buffer stores (PPO recomputes its log-prob later from this).
    """

    # No gradients during collection — we are only generating data here.
    with torch.no_grad():
        observation = _observation_to_tensor(obs, device)
        features = encoder(observation)
        mean, std = actor(features)
        value = critic(features)

        # Sample a raw (pre-squash) action and its tanh-corrected, finite log-prob.
        raw_action, log_prob = sample_action(mean, std, torch_generator)

        # Turn the raw action into a legal environment action: squash+rescale onto
        # [low, high], then clamp to mop up any floating-point overshoot.
        rescaled = tanh_rescale(raw_action, low, high)
        env_action_tensor = clamp_to_bounds(rescaled, low, high)

    # Strip the batch dimension and move everything to plain Python / NumPy for the
    # environment step and the buffer.
    env_action = env_action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    raw_action_vector = raw_action.squeeze(0).cpu().numpy().astype(np.float32)
    return env_action, raw_action_vector, float(log_prob.item()), float(value.item())


def _critic_value(
    obs: np.ndarray,
    encoder: CNNEncoder,
    critic: CriticHead,
    device: torch.device,
) -> float:
    """Return the Critic's scalar value estimate for a single observation.

    Used for the GAE *bootstrap*: the value of the observation that follows the last
    collected step (Req 5.3). No gradients needed — this is just an estimate.
    """

    with torch.no_grad():
        features = encoder(_observation_to_tensor(obs, device))
        value = critic(features)
    return float(value.item())


def collect_rollout(
    env: gym.Env,
    current_obs: np.ndarray,
    buffer: RolloutBuffer,
    encoder: CNNEncoder,
    actor: ActorHead,
    critic: CriticHead,
    low: torch.Tensor,
    high: torch.Tensor,
    torch_generator: torch.Generator,
    device: torch.device,
    logger: TensorBoardLogger,
    episode_index: int,
    episode_reward: float,
):
    """Collect EXACTLY ``rollout_horizon`` transitions into ``buffer`` (the COLLECT beat).

    Why exactly the horizon, no more and no less? The buffer is sized to
    ``rollout_horizon`` and the on-policy update consumes precisely that many
    transitions. Collecting past it would overflow the fixed buffer (and mix in data
    the update doesn't expect); stopping short would under-fill it. So the loop runs
    until the buffer is full and then stops on the dot (Req 4.5).

    For each step we select an action on the *current* observation, step the env once,
    and store the transition keyed to the observation the action was chosen on (Req
    4.1, 4.2). Episode boundaries are handled per PPO's conventions: a true terminal is
    stored with ``terminal=True`` and the env is reset (Req 4.3); a truncation (time
    limit, truncated-but-not-terminated) is stored with ``terminal=False`` and the env
    is reset, because the episode was merely cut short, not genuinely over (Req 4.4). A
    completed episode's total reward is logged, and the episode index advances.

    An ``env.step`` failure is NOT caught — it stops collection and propagates (Req 4.7).

    Returns ``(next_obs, episode_index, episode_reward)`` so the caller can compute the
    GAE bootstrap from ``next_obs`` and carry the running episode bookkeeping forward.
    """

    obs = current_obs

    # Fill the buffer to exactly the horizon, then stop (Req 4.5). len(buffer) grows
    # by one per stored transition, so this condition halts precisely at the horizon.
    while len(buffer) < buffer.rollout_horizon:
        # Pick an action on the observation we are about to store it against.
        env_action, raw_action, log_prob, value = _select_action(
            obs, encoder, actor, critic, low, high, torch_generator, device
        )

        # Take exactly one environment step. A failure here propagates (Req 4.7).
        next_obs, reward, terminated, truncated, _info = env.step(env_action)
        episode_reward += float(reward)

        # Store the transition keyed to the observation the action was selected on.
        # The terminal flag is True ONLY for a genuine terminal step; a truncation is
        # stored as non-terminal so GAE still bootstraps across the cut (Req 4.3, 4.4).
        buffer.add(
            obs=obs,
            action=raw_action,
            log_prob=log_prob,
            reward=float(reward),
            terminal=bool(terminated),
            value=value,
        )

        if terminated or truncated:
            # Episode ended (either genuinely or by truncation): log its total reward,
            # advance the episode index, and reset to start a fresh episode. Collection
            # continues into the SAME buffer until the horizon is reached (Req 4.3, 4.4).
            logger.log_episode_reward(episode_index, episode_reward)
            episode_index += 1
            episode_reward = 0.0
            obs, _info = env.reset()
        else:
            # Ordinary step: carry the next observation forward.
            obs = next_obs

    # ``obs`` is now the observation that FOLLOWS the last collected step — exactly
    # what the GAE bootstrap needs.
    return obs, episode_index, episode_reward


def train(config: Config) -> TrainResult:
    """Train the PPO agent end to end: build, then loop collect -> GAE -> update -> clear.

    This is the capstone a learner reads last. It wires every module together and runs
    the on-policy training loop until the configured step budget is spent (Req 7.1-7.4),
    then saves a checkpoint of the trained networks. It returns the pieces the entry
    point needs to run evaluation; it does NOT close the logger (the entry point does
    that after evaluation).
    """

    # --- Build the environment and read its spec --------------------------
    # The training env renders nothing (fast, windowless). We read the observation
    # shape, action dimensionality, and action bounds from it (Req 1.1, 1.2, 1.3).
    env = make_env(render_mode=None)
    spec = read_env_spec(env)

    # --- Logging + device -------------------------------------------------
    logger = TensorBoardLogger(config.log_dir)
    device = select_device()
    # Record which device the run actually used, exactly once (Req 12.5).
    logger.log_device(str(device))

    # --- Build the networks on the device ---------------------------------
    # The shared CNN encoder's feature_dim (default 256) sizes both heads. Moving all
    # three onto the device keeps every operand of every op on one device (Req 2.5).
    encoder = CNNEncoder()
    actor = ActorHead(encoder.feature_dim, spec.action_dim)
    critic = CriticHead(encoder.feature_dim)
    encoder, actor, critic = move_networks_to_device(device, encoder, actor, critic)

    # --- Seed everything from config.seed ---------------------------------
    # Seeds the env, the three global RNGs (network init included), and hands back the
    # two explicit generators for action sampling and minibatch shuffling (Req 13.2).
    generators = seed_everything(config.seed, env, device)

    # --- Buffer + updater -------------------------------------------------
    # The single-use rollout buffer, sized to exactly rollout_horizon. The PPO updater
    # shares the numpy generator so its minibatch shuffling is reproducible too.
    buffer = RolloutBuffer(
        config.rollout_horizon, spec.action_dim, observation_shape=spec.observation_shape
    )
    updater = PPOUpdater(
        encoder, actor, critic, config, device, rng=generators.numpy_generator
    )

    # Action bounds as device tensors, for the per-step rescale/clamp.
    low = torch.as_tensor(spec.action_low, device=device)
    high = torch.as_tensor(spec.action_high, device=device)

    # --- The training loop ------------------------------------------------
    # Cumulative interaction-step count starts at 0 before any collection (Req 7.1).
    step_count = 0
    update_phase_index = 0
    episode_index = 0
    episode_reward = 0.0

    # Get the first observation to act on. A failed reset propagates (Req 1.6, 1.7).
    obs, _info = env.reset()

    # Loop until the step budget is spent. Because each iteration adds exactly
    # rollout_horizon steps, once the count reaches total_steps the while condition is
    # false and NO further collect/update phase begins (Req 7.4).
    while step_count < config.total_steps:
        # 1) COLLECT exactly one horizon of fresh, on-policy experience (Req 4.1-4.5).
        obs, episode_index, episode_reward = collect_rollout(
            env,
            obs,
            buffer,
            encoder,
            actor,
            critic,
            low,
            high,
            generators.torch_generator,
            device,
            logger,
            episode_index,
            episode_reward,
        )

        # 2) GAE: turn the rollout into advantages + returns. The bootstrap is the
        # Critic value of the observation that FOLLOWS the last collected step (Req
        # 5.3). If that last step was terminal, GAE's terminal mask already zeroes the
        # bootstrap, so this value is harmless in that case.
        bootstrap_value = _critic_value(obs, encoder, critic, device)
        advantages, returns = buffer.compute_gae(
            config.gamma, config.gae_lambda, bootstrap_value
        )
        advantages = normalize_advantages(advantages)

        # 3) UPDATE: exactly one update phase over this rollout (Req 7.2). Log the
        # three metrics under the update-phase index (starts at 0, Req 8.2).
        metrics = updater.update(buffer, advantages, returns)
        logger.log_update(
            update_phase_index,
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
        )
        update_phase_index += 1

        # 4) Account for the steps just taken, then CLEAR the buffer. Clearing after
        # every update is the on-policy rule: PPO's math is only valid for data from
        # the current policy, so last iteration's transitions must never be reused
        # (Req 7.3, 4.6).
        step_count += len(buffer)
        buffer.clear()

    # --- Persist the trained networks -------------------------------------
    # Training is done: save a checkpoint of all three networks (Req 10.1 wiring).
    save_checkpoint(config.checkpoint_path, encoder, actor, critic)

    # Hand the pieces back to the entry point for evaluation. The logger is NOT closed
    # here — the entry point closes it after the evaluation pass.
    return TrainResult(
        encoder=encoder,
        actor=actor,
        critic=critic,
        action_low=low,
        action_high=high,
        logger=logger,
        device=device,
    )


# ---------------------------------------------------------------------------
# The deterministic evaluation pass (with headless video)
# ---------------------------------------------------------------------------
#
# Evaluation vs. training: the one thing that changes is exploration
# ------------------------------------------------------------------
# The training loop *samples* actions from the Gaussian the Actor produces — that
# random wiggle is exploration, and it is how PPO discovers good driving in the first
# place. Evaluation asks a different question: "now that training is done, how well
# does the policy actually drive?" To answer that cleanly we turn exploration OFF and
# always take the policy's single best guess. That is exactly what
# :func:`policy.deterministic_action` does: it squashes the Actor *mean* onto the
# action bounds with no sampled noise (Req 3.4, 11.1). Reusing that one function here
# means a learner sees a single, named "exploration off" code path rather than a second
# copy of the squash-and-rescale logic.
#
# Why record video here, and why it works without a screen
# ---------------------------------------------------------
# Watching the agent drive is the most satisfying part of the whole project, so the
# evaluation pass records at least one episode to a video file using Gymnasium's
# BUILT-IN recorder, :class:`gymnasium.wrappers.RecordVideo` (Req 9.1, 9.5). The key
# detail for a hosted box like spark.enverge.ai is that the environment renders with
# ``render_mode="rgb_array"``: frames are produced as in-memory NumPy arrays and written
# straight to an ``.mp4``/image sequence, so NO on-screen window and NO display server
# are ever needed (Req 9.1). The recorder only captures video; it never changes how
# actions are chosen.
#
# Why evaluate() builds its OWN environment
# -----------------------------------------
# The ``env`` :func:`train` created is windowless (``render_mode=None``) and so renders
# nothing — it cannot be recorded. Rather than reuse it, :func:`evaluate` constructs its
# own evaluation environment with ``render_mode="rgb_array"`` and wraps it in
# ``RecordVideo``, so evaluate fully controls both the render mode and the recorder
# (Req 9.1). The windowless training ``env`` is intentionally NOT used for the recorded
# run.
#
# Graceful degradation: a video problem never aborts the measurement
# -------------------------------------------------------------------
# Recording is a nice-to-have; the reward number is the point. So every video-related
# failure degrades gracefully instead of crashing the pass. If the video directory can't
# be created or the recorder can't be constructed, evaluate prints an error indication
# naming the directory and falls back to a plain (unrecorded) ``rgb_array`` env so the
# episodes still run and a mean reward is still reported (Req 9.3, 9.4, 9.6). A failure
# while stepping a recorded episode (e.g. a video *write* error) is likewise caught so it
# cannot abort the remaining episodes (Req 9.4). Either way, :func:`evaluate` always
# returns the mean total reward.


def _deterministic_env_action(
    obs: np.ndarray,
    encoder: CNNEncoder,
    actor: ActorHead,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Pick the single, noise-free evaluation action for ``obs`` (exploration OFF).

    This is the evaluation twin of :func:`_select_action`, but with the randomness
    removed. It runs the shared encoder once, asks the Actor for its Gaussian
    ``(mean, std)``, and then takes ONLY the ``mean`` — discarding ``std`` entirely
    because we add no sampled noise during evaluation. The mean is turned into a legal
    environment action by :func:`policy.deterministic_action` (tanh-squash + rescale,
    no noise) and clamped to mop up any floating-point overshoot (Req 3.4, 3.5, 11.1).

    Returns a NumPy action vector ready for ``env.step``.
    """

    # No gradients during evaluation — we are only running the policy forward.
    with torch.no_grad():
        observation = _observation_to_tensor(obs, device)
        features = encoder(observation)
        mean, _std = actor(features)  # std deliberately unused: exploration is OFF.

        # The single "exploration off" path: squash+rescale the MEAN onto [low, high],
        # then clamp. No Gaussian sample is drawn anywhere here (Req 11.1).
        action_tensor = deterministic_action(mean, low, high)
        action_tensor = clamp_to_bounds(action_tensor, low, high)

    # Strip the batch dimension and hand back a plain NumPy vector for env.step.
    return action_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def _make_recording_env(video_dir: str):
    """Build an ``rgb_array`` eval env, wrapped in ``RecordVideo`` when possible.

    Returns ``(env, error)`` where ``error`` is ``None`` on success or a short,
    human-readable error string if the video directory could not be created or the
    recorder could not be constructed. On failure the returned ``env`` is a plain
    (unrecorded) ``rgb_array`` environment so evaluation can still run and still report
    a mean reward (Req 9.3, 9.4, 9.6) — graceful degradation, never an abort.

    The recorder uses ``episode_trigger=lambda ep: ep == 0`` so AT LEAST the first
    episode is captured, satisfying "record at least one episode" (Req 9.5). Because the
    env renders with ``render_mode="rgb_array"``, recording needs no on-screen window or
    display server, which is what makes it work on a headless host (Req 9.1).
    """

    # The env that will actually be driven renders to rgb_array frames (no window).
    eval_env = make_env(render_mode="rgb_array")

    try:
        # Create the output directory (including any missing parents) before wrapping,
        # so the recorder has somewhere to write (Req 9.3).
        os.makedirs(video_dir, exist_ok=True)

        # Gymnasium's built-in recorder. episode_trigger records at least episode 0,
        # guaranteeing one recorded episode (Req 9.1, 9.5).
        recorded_env = RecordVideo(
            eval_env,
            video_folder=video_dir,
            episode_trigger=lambda episode_index: episode_index == 0,
        )
        return recorded_env, None
    except Exception as error:
        # Directory creation or recorder construction failed. Do NOT abort the pass:
        # report the problem (naming the directory) and fall back to the plain,
        # unrecorded rgb_array env so evaluation still runs (Req 9.4, 9.6).
        message = (
            f"Video recording disabled: could not set up RecordVideo in directory "
            f"'{video_dir}' ({error}). Evaluation will continue without a recording."
        )
        print(message)
        return eval_env, message


def evaluate(config: Config, nets: TrainResult) -> float:
    """Run the deterministic, exploration-OFF evaluation pass and return the mean reward.

    ``nets`` is the :class:`TrainResult` that :func:`train` returns; evaluate reads the
    trained ``encoder``/``actor``/``critic``, the action bounds (``action_low`` /
    ``action_high``), the ``device``, and the ``logger`` from it. (The Critic is not
    needed to *choose* actions during evaluation — action selection is Actor-mean only —
    so it is simply not used here.)

    What this pass does, step by step:

    1. Build its OWN evaluation environment with ``render_mode="rgb_array"`` wrapped in
       :class:`gymnasium.wrappers.RecordVideo`, so it controls rendering and recording
       and captures at least the first episode headlessly (Req 9.1, 9.5). The windowless
       training env is deliberately not reused. If the video setup fails, it degrades to
       a plain unrecorded env and keeps going (Req 9.3, 9.4, 9.6).
    2. Run ``config.eval_episodes`` episodes. (That value is already validated to be in
       [1, 10000] by :meth:`Config.validate` per Req 11.3, so it is not re-checked here.)
       Each episode resets the env and steps until ``terminated`` OR ``truncated``,
       choosing every action DETERMINISTICALLY from the Actor mean via
       :func:`policy.deterministic_action` — no sampled Gaussian noise (Req 11.1). The
       per-episode total reward is the sum of the per-step rewards (Req 11.4).
    3. Report the arithmetic MEAN total reward across all episodes (Req 11.5) and log it
       as a single scalar via ``logger.log_eval_reward`` (Req 11.6).

    Returns the mean total reward as a ``float``.
    """

    encoder = nets.encoder
    actor = nets.actor
    low = nets.action_low
    high = nets.action_high
    device = nets.device
    logger = nets.logger

    # Step 1: build the recording env (falls back to a plain env on any video failure).
    eval_env, _video_error = _make_recording_env(config.video_dir)

    # Step 2: run eval_episodes deterministic episodes, accumulating per-episode reward.
    # eval_episodes is already validated to be in [1, 10000] by Config.validate (Req 11.3).
    episode_total_rewards: list[float] = []
    try:
        for _episode in range(config.eval_episodes):
            obs, _info = eval_env.reset()
            episode_reward = 0.0

            # Step until the episode genuinely ends OR is truncated (Req 11.4).
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = _deterministic_env_action(
                    obs, encoder, actor, low, high, device
                )
                obs, reward, terminated, truncated, _info = eval_env.step(action)
                episode_reward += float(reward)

            episode_total_rewards.append(episode_reward)
    except Exception as error:
        # A failure while stepping a RECORDED episode (e.g. a video write error) must
        # not abort the whole pass. Report it and continue to reporting the mean over
        # whatever episodes did complete (Req 9.4). If nothing completed, we fall back
        # to a mean of 0.0 below so a finite value is still returned/logged.
        print(
            f"Evaluation episode interrupted ({error}); reporting the mean over "
            f"{len(episode_total_rewards)} completed episode(s)."
        )
    finally:
        # Close the eval env (flushes/finalizes any recorded video). Closing is best
        # effort — a close failure must not break the returned measurement.
        try:
            eval_env.close()
        except Exception as error:
            print(f"Could not cleanly close the evaluation environment ({error}).")

    # Step 3: arithmetic mean total reward across all completed episodes (Req 11.5).
    if episode_total_rewards:
        mean_reward = sum(episode_total_rewards) / len(episode_total_rewards)
    else:
        # No episode completed (only possible if every recorded episode errored). Report
        # a finite 0.0 so the pass still yields a usable, loggable number.
        mean_reward = 0.0

    # Log the single evaluation scalar (Req 11.6). The logger degrades gracefully on its
    # own, so we do not need to guard this call.
    logger.log_eval_reward(mean_reward)

    return mean_reward


# ---------------------------------------------------------------------------
# Artifact path anchoring
# ---------------------------------------------------------------------------
#
# Why artifacts are anchored to the project root, not the working directory
# -------------------------------------------------------------------------
# The ``Config`` artifact paths (``log_dir``, ``video_dir``, ``checkpoint_path``) are
# written as *relative* strings like ``artifacts/...``. A relative path resolves against
# whatever directory you launched Python from, so running ``python main.py`` from inside
# ``src/`` would scatter an ``artifacts/`` folder under ``src/``. We want a single,
# predictable location: ``3_PPO/artifacts/`` — the project root, one level above this
# file — no matter where the program is launched from. :func:`anchor_artifact_paths`
# rewrites any relative artifact path so it is anchored there.

# The project root is the directory that contains the ``src`` package — i.e. the parent
# of the directory holding this file (``3_PPO/``).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def anchor_artifact_paths(config: Config) -> Config:
    """Return a copy of ``config`` with artifact paths anchored to the project root.

    Each of ``log_dir``, ``video_dir``, and ``checkpoint_path`` is resolved against the
    project root (``3_PPO/``) when it is relative, so artifacts always land in
    ``3_PPO/artifacts/`` regardless of the current working directory. Absolute paths are
    left untouched, so a learner who passes an explicit absolute location still gets it.
    """

    def _anchor(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)

    return replace(
        config,
        log_dir=_anchor(config.log_dir),
        video_dir=_anchor(config.video_dir),
        checkpoint_path=_anchor(config.checkpoint_path),
    )


# ---------------------------------------------------------------------------
# The in-code entry point
# ---------------------------------------------------------------------------
#
# Why there is no command line here
# ---------------------------------
# This is the file a learner reads *last*, and it should read like a short story of the
# whole run. Everything is configured IN CODE: you build a :class:`Config` right here,
# with any experiment tweaks written as plain literals. There is deliberately NO
# argparse and NO CLI parsing of any kind (Req 13.1, 13.6) — to change a hyperparameter
# you edit the ``Config(...)`` below, not a shell command. That keeps the entire program
# traceable from this one function.
#
# The order of the steps is the whole point
# ------------------------------------------
# ``main`` is a straight top-to-bottom narration: build the config, VALIDATE it first so
# a bad knob is caught before any env/network/file is created, train (which internally
# selects the device, builds the env/networks/buffer, runs the collect -> GAE -> update
# -> clear loop, and saves a checkpoint — Req 7.1, 10.1), then run the deterministic
# evaluation pass. The logger is closed in a ``finally`` so its metrics are always
# flushed to disk, even if evaluation raises.


def main() -> None:
    """Run the whole PPO CarRacing program end to end, configured entirely in code.

    Read this as a linear narration of one full run:

    1. **Build the config in code.** Construct a :class:`Config` directly — defaults are
       sensible, so ``Config()`` is enough; to experiment, pass literal overrides such as
       ``Config(total_steps=200_000, seed=0)``. No command line is parsed (Req 13.1, 13.6).
    2. **Validate first.** Call ``config.validate()`` *before* constructing anything else,
       so a bad hyperparameter is reported (by field name) before any environment,
       network, or file is created.
    3. **Train.** ``train(config)`` selects the device, builds the env/networks/buffer,
       runs the on-policy loop, and saves a checkpoint (Req 7.1, 10.1). It returns a
       :class:`TrainResult` bundle but does NOT close the logger.
    4. **Evaluate.** ``evaluate(config, result)`` runs the deterministic (exploration-off)
       pass and returns the mean total reward.
    5. **Always close the logger.** The ``finally`` guarantees ``result.logger.close()``
       runs so buffered metrics are flushed even if evaluation raises.
    """

    # Step 1: build the Config directly in code (no argparse). Defaults are fine for a
    # first run; to tweak an experiment, pass literal overrides here, e.g.:
    #     config = Config(total_steps=200_000, seed=0)
    config = Config()
    # FOR LOCAL DEVICE TESTING
    # config = Config(
    #     total_steps=20_000,      # default is 1_000_000 — far too long on CPU
    #     rollout_horizon=1024,    # smaller single-use batch = less RAM per update
    #     minibatch_size=64,
    #     update_epochs=4,         # default 10; fewer passes = faster iterations
    #     eval_episodes=2,         # default 5
    #     seed=0,                  # reproducible run
    # )


    # Anchor artifact paths to the project root (3_PPO/) so outputs always land in
    # 3_PPO/artifacts/ — never under src/ — no matter where the program is launched.
    config = anchor_artifact_paths(config)

    # Step 2: validate BEFORE creating any resource, so a bad knob fails fast and names
    # the offending field rather than blowing up mid-run.
    config.validate()

    print("Starting PPO training on CarRacing-v3...")

    # Step 3: train end to end. This internally selects the device, constructs the
    # env/networks/buffer, runs collect -> GAE -> update -> clear until the step budget
    # is spent, and saves a checkpoint (Req 7.1, 10.1). It returns the pieces evaluation
    # needs, with the logger still open.
    result = train(config)

    try:
        # Step 4: run the deterministic evaluation pass and report the mean reward.
        mean_reward = evaluate(config, result)
        print(f"Finished. Mean evaluation reward over "
              f"{config.eval_episodes} episode(s): {mean_reward:.2f}")
    finally:
        # Step 5: always close the logger so its metrics are flushed/closed, even if the
        # evaluation pass raised above.
        result.logger.close()


if __name__ == "__main__":
    main()
