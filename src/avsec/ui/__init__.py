"""Local web interface for the avsec testbed.

It is a thin front end over the same library the CLI uses: no experiment logic
lives here.  Implemented on the standard library ``http.server`` so the project
needs no web framework dependency.
"""

from avsec.ui.server import serve  # noqa: F401

__all__ = ["serve"]
