"""avsec - research testbed for secure video over an existing analog video path.

Public API is intentionally small and stable; the CLI and the web UI both use it.
"""

__version__ = "0.1.0"

from avsec.utils import Stopwatch, StageTimer, sha256_file  # noqa: F401

__all__ = ["__version__", "Stopwatch", "StageTimer", "sha256_file"]
