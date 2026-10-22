"""Pytest configuration for the PPO test package.

Puts the ``3_PPO`` package directory on ``sys.path`` so the tests can import the
policy/networks/buffer modules directly (``import policy``) regardless of the
directory pytest is invoked from.
"""

import os
import sys

_PPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PPO_DIR not in sys.path:
    sys.path.insert(0, _PPO_DIR)
