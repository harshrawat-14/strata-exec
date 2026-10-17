"""
Regenerate the golden fixtures from the CURRENT binary and code.

Run from the repo root, only on a commit whose behaviour is known-good:
    cargo build --release && python rl/tests/golden/generate.py

Regenerating after a behaviour change silently blesses that change -- if a
refactor makes test_golden_regression.py fail, fix the refactor, not the
fixture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # rl/

from environment import RL_ENV_BIN  # noqa: E402
import golden_lib  # noqa: E402


def main() -> None:
    env = golden_lib.collect_env_golden(RL_ENV_BIN)
    golden_lib.ENV_FIXTURE.write_text(json.dumps(env, indent=1, sort_keys=True) + "\n")
    print(f"wrote {golden_lib.ENV_FIXTURE}")

    ev = golden_lib.collect_eval_golden()
    golden_lib.EVAL_FIXTURE.write_text(json.dumps(ev, indent=1, sort_keys=True) + "\n")
    print(f"wrote {golden_lib.EVAL_FIXTURE}")


if __name__ == "__main__":
    main()
