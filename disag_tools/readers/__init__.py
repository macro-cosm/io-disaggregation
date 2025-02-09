"""
Readers module for handling various input-output table formats.

This module provides readers for different types of input-output tables,
with standardized interfaces for loading and preprocessing the data.
"""

from disag_tools.readers.icio_reader import ICIOReader
from disag_tools.readers.readers import IOTableReader

__all__ = ["IOTableReader", "ICIOReader"]
