"""Pytest configuration for the DQN test package.

Puts the ``2_DQN`` package directory on ``sys.path`` so the tests can import the
component modules directly (``import config``, ``import networks``, ...) the same
way those modules import each other (e.g. ``policy.py`` does ``from config import
Config``). This works regardless of the directory pytest is invoked from.
"""

import os
import sys

_DQN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DQN_DIR not in sys.path:
    sys.path.insert(0, _DQN_DIR)
