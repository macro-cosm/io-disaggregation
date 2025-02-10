"""Module for constructing target vectors in disaggregation problems."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from disag_tools.configurations.config import DisaggregationConfig
from disag_tools.readers.blocks import DisaggregationBlocks, SectorId, SectorInfo

logger = logging.getLogger(__name__)


@dataclass
class DisaggregationTargets:
    """
    Class to construct target vectors for disaggregation problems.

    For each sector n being disaggregated, we need to construct a target vector Y^n
    that contains:
    - B^n: The original flows from undisaggregated sectors to sector n
    - C^n: The original flows from sector n to undisaggregated sectors
    - D^n: The original flows between sector n and other disaggregated sectors
    - w^n: The relative output weights for sector n's subsectors

    These vectors are used in the constraint equations:
    M^n X^n = Y^n

    Attributes:
        blocks: The DisaggregationBlocks instance containing the matrix structure
        config: The DisaggregationConfig containing the weights
    """

    blocks: DisaggregationBlocks
    config: DisaggregationConfig

    def get_weights(self, sector_info: SectorInfo) -> np.ndarray:
        """
        Get the relative output weights for a sector's subsectors.

        Args:
            sector_info: Information about the sector

        Returns:
            Array of weights, one per subsector
        """
        if not self.config.sectors:
            raise ValueError("No sector disaggregation configuration provided")

        # Get the sector configuration
        sector = sector_info.sector
        if sector not in self.config.sectors:
            raise ValueError(f"No configuration found for sector {sector}")

        sector_config = self.config.sectors[sector]

        # For multi-region case, we need the country's weights
        if sector_info.is_multi_region:
            country = sector_info.country
            if not country:
                raise ValueError(f"No country found for multi-region sector {sector}")

            # Get weights for this country
            weights = []
            for subsector in sector_config.subsectors.values():
                if country not in subsector.relative_output_weights:
                    raise ValueError(f"No weight found for country {country} in sector {sector}")
                weights.append(subsector.relative_output_weights[country])

        else:
            # For single-region case, just get the weights in order
            weights = [
                subsector.relative_output_weights.get("", 0.0)  # Empty string for single region
                for subsector in sector_config.subsectors.values()
            ]

        # Validate that we got the right number of weights
        if len(weights) != sector_info.k:
            raise ValueError(
                f"Expected {sector_info.k} weights for sector {sector}, got {len(weights)}"
            )

        return np.array(weights)

    def get_target_vector(self, n: int) -> np.ndarray:
        """
        Construct the target vector Y^n for sector n.

        The vector Y^n contains:
        1. B^n: Original flows from undisaggregated to sector n
        2. C^n: Original flows from sector n to undisaggregated
        3. D^n: Original flows between n and other disaggregated sectors
        4. w^n: Relative output weights for sector n's subsectors

        Args:
            n: Index of the sector (1-based as in math notation)

        Returns:
            The target vector Y^n
        """
        if not 1 <= n <= self.blocks.K:
            raise ValueError(f"Sector index {n} out of range [1, {self.blocks.K}]")

        # Get sector info
        sector_info = self.blocks.get_sector_info(n)

        # Get the blocks
        B = self.blocks.get_B(n).flatten()  # Flows from undisaggregated to n
        C = self.blocks.get_C(n).flatten()  # Flows from n to undisaggregated

        # Get flows between n and other disaggregated sectors
        D = []
        for l in range(1, self.blocks.K + 1):
            D.append(self.blocks.get_D(n, l).flatten())
        D = np.concatenate(D)

        # Get weights for this sector
        w = self.get_weights(sector_info)

        # Concatenate all components
        Y = np.concatenate([B, C, D, w])

        logger.debug(f"Constructed target vector for sector {n}")
        logger.debug(f"B shape: {B.shape}, C shape: {C.shape}")
        logger.debug(f"D shape: {D.shape}, w shape: {w.shape}")
        logger.debug(f"Final Y shape: {Y.shape}")

        return Y

    def get_target_vector_by_sector_id(self, sector_id: SectorId) -> np.ndarray:
        """
        Construct the target vector Y^n for a sector identified by its ID.

        The vector Y^n contains:
        1. B^n: Original flows from undisaggregated to sector n
        2. C^n: Original flows from sector n to undisaggregated
        3. D^n: Original flows between n and other disaggregated sectors
        4. w^n: Relative output weights for sector n's subsectors

        Args:
            sector_id: The sector identifier (str for single-region, tuple[str, str] for multi-region)

        Returns:
            The target vector Y^n

        Raises:
            ValueError: If sector_id is not found in the blocks
        """
        # Find the sector index
        for n, sector_info in enumerate(self.blocks.sectors, start=1):
            if sector_info.sector_id == sector_id:
                return self.get_target_vector(n)

        raise ValueError(f"Sector {sector_id} not found in blocks")
