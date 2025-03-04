"""Module for handling disaggregation problem formulation and solution."""

import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from disag_tools.configurations.config import DisaggregationConfig

from disag_tools.readers.disaggregation_blocks import (
    SectorId,
    SectorInfo,
    DisaggregationBlocks,
    unfold_countries,
)
from disag_tools.readers.icio_reader import ICIOReader

logger = logging.getLogger(__name__)

# Type aliases for clarity
Array: TypeAlias = np.ndarray


@dataclass
class SectorDisaggregationProblem:
    """Represents a single sector's disaggregation problem.

    This class handles the construction and solution of the disaggregation problem
    for a single sector, including:
    - Matrix generation (M^n)
    - Target vector construction (Y^n)
    - Prior information handling
    - Problem setup and solution

    """

    aggregated_sector: SectorInfo
    disaggregated_sectors: list[SectorId]
    m_matrix: Array
    y_vector: Array

    @classmethod
    def from_disaggregation_block(
        cls,
        block: DisaggregationBlocks,
        n: int,
        weights_list: list[Array],
        aggregated_sector: SectorInfo,
        disaggregated_sectors: list[SectorId],
    ):
        m_matrix = block.get_large_m(n, weights_list)
        y_vector = block.get_y_vector(n, weights_list[n - 1])

        return cls(
            aggregated_sector=aggregated_sector,
            disaggregated_sectors=disaggregated_sectors,
            m_matrix=m_matrix,
            y_vector=y_vector,
        )


@dataclass
class DisaggregationProblem:
    problems: list[SectorDisaggregationProblem]
    disaggregation_blocks: DisaggregationBlocks
    weights: list[Array]

    @classmethod
    def from_configuration(cls, config: DisaggregationConfig, reader: ICIOReader):
        mapping = config.get_simplified_mapping()

        sectors_info = unfold_countries(reader.countries, mapping)

        # setup blocks
        blocks = DisaggregationBlocks.from_technical_coefficients(
            tech_coef=reader.technical_coefficients,
            sectors_info=sectors_info,
            output=reader.output_from_out,
        )

        weight_dict = config.get_weight_dictionary()

        disag_mapping = config.get_disagg_mapping()

        weights = []

        for sector in blocks.sectors:
            subsectors = disag_mapping[sector.sector_id]
            weights.append(np.array([weight_dict[subsector] for subsector in subsectors]))

        problems = []
        for sector in blocks.sectors:
            subsectors = disag_mapping[sector.sector_id]
            problems.append(
                SectorDisaggregationProblem.from_disaggregation_block(
                    blocks, sector.index, weights, sector, list(subsectors)
                )
            )

        return cls(problems=problems, disaggregation_blocks=blocks, weights=weights)
