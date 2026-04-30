"""Backward-compatible import shim for the historical ``sas`` namespace.

New code should import from :mod:`suma`.  The shim keeps older notebooks,
scripts, and tests working while the project-facing package name moves to SUMA.
"""

from __future__ import annotations

from importlib import import_module

_suma = import_module("suma")

__path__ = _suma.__path__
__all__ = getattr(_suma, "__all__", [])
__version__ = getattr(_suma, "__version__", "0.3.0")

PROJECT_ROOT = _suma.PROJECT_ROOT
CONFIGS_DIR = _suma.CONFIGS_DIR
DEFAULT_CONFIG_PATH = _suma.DEFAULT_CONFIG_PATH
