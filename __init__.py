"""
Numera
=======

AI-powered mathematics visualization engine.

Transforms mathematical expressions into animated visualizations
using symbolic mathematics, LLM reasoning, and Manim.

Author: Mubashir Shah
License: GPL-3.0
"""

__version__ = "0.1.0"
__author__ = "Mubashir Shah"
__license__ = "GPL-3.0"

# Public API
from .pipeline import process_job

__all__ = [
    "process_job",
]
