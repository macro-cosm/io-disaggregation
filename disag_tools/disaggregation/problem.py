"""Module for handling disaggregation problem formulation and solution."""

import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from disag_tools.configurations.config import DisaggregationConfig
from disag_tools.disaggregation.disaggregation_blocks import (
    DisaggregationBlocks,
    SectorId,
    SectorInfo,
    unfold_countries,
)
from disag_tools.disaggregation.solution_blocks import SolutionBlocks
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
    """Class representing the complete disaggregation problem.

    This class contains both the optimization problems for each sector and the solution
    structure that will hold the results.

    Attributes:
        problems: List of individual sector disaggregation problems
        disaggregation_blocks: Original blocks containing the aggregated data
        solution_blocks: Structure that will hold the disaggregated solution
        weights: List of weight arrays for each sector being disaggregated
    """

    problems: list[SectorDisaggregationProblem]
    disaggregation_blocks: DisaggregationBlocks
    solution_blocks: SolutionBlocks
    weights: list[Array]

    @classmethod
    def from_configuration(cls, config: DisaggregationConfig, reader: ICIOReader):
        """Create a DisaggregationProblem from a configuration and reader.

        Args:
            config: Configuration specifying the disaggregation structure
            reader: Reader containing the input-output data

        Returns:
            DisaggregationProblem instance containing both the problems to solve
            and the structure to hold the solution
        """
        mapping = config.get_simplified_mapping()
        disag_mapping = config.get_disagg_mapping()
        weight_dict = config.get_weight_dictionary()

        # Setup the disaggregation blocks
        sectors_info = unfold_countries(reader.countries, mapping)
        blocks = DisaggregationBlocks.from_technical_coefficients(
            tech_coef=reader.technical_coefficients,
            sectors_info=sectors_info,
            output=reader.output_from_out,
        )

        # Create the solution blocks structure
        solution = SolutionBlocks.from_disaggregation_blocks(blocks, disag_mapping)

        # Create weights list for each sector being disaggregated
        weights = []
        for sector in blocks.sectors:
            subsectors = disag_mapping[sector.sector_id]
            weights.append(np.array([weight_dict[subsector] for subsector in subsectors]))

        # Create individual problems for each sector
        problems = []
        for sector in blocks.sectors:
            subsectors = disag_mapping[sector.sector_id]
            problems.append(
                SectorDisaggregationProblem.from_disaggregation_block(
                    blocks, sector.index, weights, sector, sorted(list(subsectors))
                )
            )

        return cls(
            problems=problems,
            disaggregation_blocks=blocks,
            solution_blocks=solution,
            weights=weights,
        )
