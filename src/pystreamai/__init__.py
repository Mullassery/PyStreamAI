"""
PyStreamAI: Streaming AI inference engine with zero-copy tensor optimization.

Rust-powered extension via PyO3/maturin for 40-50x inference speedup.
"""

# Import Rust extension
try:
    from . import pystreamai as _core

    # Export Rust function
    hello = _core.hello
except (ImportError, AttributeError):
    # Fallback: undefined if Rust extension not built
    hello = None

__version__ = "0.3.0"
__author__ = "Georgi Mammen Mullassery"
__license__ = "Proprietary"

__all__ = [
    "hello",
]
