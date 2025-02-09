"""
Protocol definitions for input-output table readers.

This module defines the interface that all IO table readers must implement,
ensuring consistent access to input-output data regardless of the source format.
"""

from typing import Protocol

import pandas as pd


class IOTableReader(Protocol):
    """
    Protocol defining the interface for input-output table readers.

    This protocol ensures that all IO table readers provide consistent access
    to the underlying data, regardless of the source format (ICIO, WIOD, etc.).

    Properties:
        get_iot: Returns the input-output transactions matrix
        get_outputs: Returns the total output vector
    """

    @property
    def get_iot(self) -> pd.DataFrame:
        """
        Get the input-output transactions matrix.

        The matrix should be in a standardized format with MultiIndex on both
        rows and columns, where each index level represents (country, industry).

        Returns:
            pd.DataFrame: The input-output transactions matrix
        """
        ...

    @property
    def get_outputs(self) -> pd.Series:
        """
        Get the total output vector.

        The series should have a MultiIndex where each level represents
        (country, industry), matching the structure of the IO table.

        Returns:
            pd.Series: The total output vector
        """
        ...
