"""
Package distribution related utils.
"""

import importlib

from typing import Optional

PACKAGE_NAME = "gordo-core"


def get_version() -> Optional[str]:
    """
    Get the current gordo-core version. ``None`` if not installed in the system.
    """
    try:
        return importlib.metadata.version("gordo-core")
    except ImportError:
        pass
    return None
