"""Module for handling bottom blocks (VA and TLS) in disaggregation problems."""

import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from disag_tools.readers.icio_reader import ICIOReader

logger = logging.getLogger(__name__)

# Type alias for sector identifiers
SectorId: TypeAlias = str | tuple[str, str]


@dataclass
class BottomBlocks:
    """Class to handle the bottom blocks (VA and TLS) of a disaggregation problem.

    This class manages the Value Added (VA) and Taxes Less Subsidies (TLS) rows
    of the input-output table. When sectors are disaggregated:
    - VA is allocated proportionally to output weights
    - TLS is left as NaN to be computed later in the assembler

    Attributes:
        data: DataFrame containing VA and TLS rows for all sectors
    """

    data: pd.DataFrame

    @classmethod
    def from_disaggregation_blocks(
        cls,
        reader: ICIOReader,
        disagg_mapping: dict[SectorId, list[SectorId]],
        weight_dict: dict[SectorId, float],
    ) -> "BottomBlocks":
        """Create BottomBlocks from a reader and disaggregation mapping.

        Args:
            reader: ICIOReader containing the original data
            disagg_mapping: Dictionary mapping aggregated sectors to their subsectors
                e.g. {("ROW", "A"): [("ROW", "A01"), ("ROW", "A03")]}
            weight_dict: Dictionary mapping each subsector to its relative output weight
                e.g. {("ROW", "A01"): 0.6, ("ROW", "A03"): 0.4}

        Returns:
            BottomBlocks instance with VA allocated by output weights and TLS as NaN
        """
        # Create list of all sectors after disaggregation
        all_sectors = []
        for agg_sector, subsectors in disagg_mapping.items():
            # Remove aggregated sector and add its subsectors
            if agg_sector in reader.data.columns:
                all_sectors.extend(subsectors)
            else:
                all_sectors.append(agg_sector)

        # Create DataFrame with VA and TLS rows
        data = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [("VA", "VA"), ("TLS", "TLS")], names=["CountryInd", "industryInd"]
            ),
            columns=pd.MultiIndex.from_tuples(all_sectors, names=["CountryInd", "industryInd"]),
            dtype=float,
        )

        # Fill VA values using output weights for disaggregated sectors
        for agg_sector, subsectors in disagg_mapping.items():
            if agg_sector in reader.data.columns:
                va_value = reader.data.loc[("VA", "VA"), agg_sector]
                for subsector in subsectors:
                    data.loc[("VA", "VA"), subsector] = va_value * weight_dict[subsector]

        # Fill TLS values as NaN - they will be computed in the assembler
        data.loc[("TLS", "TLS"), :] = np.nan

        return cls(data=data)
