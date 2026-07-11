"""
Pytest configuration for rl/ tests.

rl/evaluate.py and rl/environment.py use bare `import` (e.g.
`from environment import StrataExecEnv`) assuming rl/ itself is on
sys.path, rather than being imported as a package (rl.evaluate). Match
that convention here so tests can import them directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent.parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))
