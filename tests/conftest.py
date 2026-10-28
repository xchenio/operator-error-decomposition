"""Make the repository root importable however the suite is invoked.

The tests import `src.*` and `scripts.*`, which only resolve when the repository root is on
sys.path. Running `python -m unittest discover -s tests` from the root happens to provide it
via the current directory; running `pytest tests/` from anywhere else does not.

pytest picks this file up automatically. For plain unittest, discovery from the repository
root (as documented in README and CI) is enough.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
