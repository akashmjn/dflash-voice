"""Put this directory on sys.path so tests can import the modules as siblings.

Not a package (no __init__.py), so ``tests/`` has no other way to reach
``model``/``train``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
