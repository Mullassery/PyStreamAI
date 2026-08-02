"""
PyStreamAI: Python package

Rust-powered extension via PyO3/maturin.
"""

try:
    from ._core import *  # noqa: F401, F403
except ImportError:
    # Fallback for development/import errors
    pass

__version__ = "0.3.0"
__all__ = ['Stream', 'AI', 'process']
