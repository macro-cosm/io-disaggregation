"""Module for handling disaggregation problem formulation and solution."""

import logging
from dataclasses import dataclass
from typing import Optional, TypeAlias

import cvxpy as cp
import numpy as np
import pandas as pd

from disag_tools.configurations.config import DisaggregationConfig
from disag_tools.disaggregation.bottom_blocks import BottomBlocks
from disag_tools.disaggregation.disaggregation_blocks import (
    DisaggregationBlocks,
    SectorInfo,
    unfold_sectors_info,
)
from disag_tools.disaggregation.final_demand_blocks import FinalDemandBlocks
from disag_tools.disaggregation.planted_solution import PlantedSolution
from disag_tools.disaggregation.prior_blocks import PriorBlocks, PriorInfo
from disag_tools.disaggregation.solution_blocks import SolutionBlocks
from disag_tools.disaggregation.utils import SectorId, _check_regional
from disag_tools.readers.icio_reader import ICIOReader

logger = logging.getLogger(__name__)

# Type aliases for clarity
Array: TypeAlias = np.ndarray


def _convert_prior_df_to_info(prior_df: pd.DataFrame) -> list[PriorInfo]:
    """Convert prior information DataFrame to list of PriorInfo tuples.

    Args:
        prior_df: DataFrame with columns:
            - For multi-country: [Country_row, Sector_row, Country_column, Sector_column, value]
            - For single-country: [Sector_row, Sector_column, value]

    Returns:
        List of (source_sector, dest_sector, value) tuples
    """
    prior_info = []

    # Check if we're dealing with multi-country or single-country format
    is_multi_country = all(col in prior_df.columns for col in ["Country_row", "Country_column"])

    if is_multi_country:
        # Multi-country case: combine country and sector into tuples
        for _, row in prior_df.iterrows():
            source = (row["Country_row"], row["Sector_row"])
            dest = (row["Country_column"], row["Sector_column"])
            prior_info.append((source, dest, row["value"]))
    else:
        # Single-country case: use sectors directly
        for _, row in prior_df.iterrows():
            prior_info.append((row["Sector_row"], row["Sector_column"], row["value"]))

    return prior_info  # type: ignore


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
    prior_vector: Array | None = None

    @classmethod
    def from_disaggregation_block(
        cls,
        block: DisaggregationBlocks,
        n: int,
        weights_list: list[Array],
        aggregated_sector: SectorInfo,
        disaggregated_sectors: list[SectorId],
        prior_blocks: PriorBlocks | None = None,
    ):
        m_matrix = block.get_large_m(n, weights_list)
        y_vector = block.get_y_vector(n, weights_list[n - 1])

        # Get prior vector if prior blocks are provided
        prior_vector = None
        if prior_blocks is not None:
            prior_vector = prior_blocks.get_prior_n_vector(n)

        return cls(
            aggregated_sector=aggregated_sector,
            disaggregated_sectors=disaggregated_sectors,
            m_matrix=m_matrix,
            y_vector=y_vector,
            prior_vector=prior_vector,
        )

    def solve(
        self,
        lambda_sparse: float = 1.0,
        mu_prior: float = 10.0,
        initial_guess: Array | None = None,
        use_initial_guess: bool = True,
    ) -> Array:
        """Solve the disaggregation problem.

        If prior information exists, solves:
            minimize    M@X = Y + λ|X[sparse]| + μ|X[known] - known|²
        Otherwise solves:
            minimize    M@X = Y

        Args:
            lambda_sparse: Weight for L1 penalty on sparse terms (default: 1.0)
            mu_prior: Weight for L2 penalty on deviation from known terms (default: 10.0)
            initial_guess: Optional initial guess for the solution vector
            use_initial_guess: Whether to use the initial guess (default: True)

        Returns:
            Solution vector X
        """
        m = self.m_matrix.shape[1]  # Number of variables
        X = cp.Variable(m)

        # Basic constraint M@X = Y and non-negativity
        constraints = [
            self.m_matrix @ X == self.y_vector,
            X >= 0,
        ]

        # Start with empty objective
        objective = 0

        if self.prior_vector is not None:
            # Get indices where we have prior information
            known_mask = ~np.isnan(self.prior_vector)

            # Add L1 penalty for terms that should be sparse (prior = 0)
            sparse_mask = known_mask & (self.prior_vector == 0)
            if np.any(sparse_mask):
                objective += lambda_sparse * cp.norm1(X[sparse_mask])

            # Add L2 penalty for deviation from known non-zero terms
            known_nonzero = known_mask & (self.prior_vector > 0)
            if np.any(known_nonzero):
                objective += mu_prior * cp.sum_squares(
                    X[known_nonzero] - self.prior_vector[known_nonzero]
                )

        # Solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)  # type: ignore

        # If we have an initial guess and want to use it, set it
        if use_initial_guess and initial_guess is not None:
            X.value = initial_guess

        prob.solve(warm_start=True)

        # if prob.status != cp.OPTIMAL:
        #     raise RuntimeError(f"Problem could not be solved optimally. Status: {prob.status}")

        solution = X.value

        if np.any(solution < 0):
            logger.warning("Problem solved but may not be optimal. Status: %s", prob.status)
            logger.debug(
                f"Number of negative values in solution vector: {np.sum(solution < 0)}",
            )
            if initial_guess is not None:
                # negative values in solution
                negative_values = solution < 0
                positive_guess = initial_guess >= 0
                working_indices = np.logical_and(negative_values, positive_guess)

                # if no working indices, raise error
                if not np.any(working_indices):
                    raise RuntimeError("No fix possible for negative values in solution.")

                # find the most negative value in the solution, but that is positive in the initial guess
                min_index = np.argmin(solution[working_indices])
                min_value = solution[working_indices][min_index]

                value_guess = initial_guess[working_indices][min_index]

                factor = value_guess / (value_guess - min_value)

                solution = factor * solution + (1 - factor) * initial_guess

                logger.debug(f"Corrected negative values in solution vector.")

            solution[solution < 0] = 0

        return solution


@dataclass
class DisaggregationProblem:
    """Class representing the complete disaggregation problem.

    This class contains both the optimization problems for each sector and the solution
    structure that will hold the results.

    Attributes:
        problems: List of individual sector disaggregation problems
        disaggregation_blocks: Original blocks containing the aggregated data
        solution_blocks: Structure that will hold the disaggregated solution
        final_demand_blocks: Structure that will hold the disaggregated final demand
        bottom_blocks: Structure that will hold the VA and TLS rows
        weights: List of weight arrays for each sector being disaggregated
        regionalised: Whether the disaggregation is regionalised (default: False)
        prior_blocks: Optional prior information for the problem
        planted_solution: Optional planted solution for testing
    """

    problems: list[SectorDisaggregationProblem]
    disaggregation_blocks: DisaggregationBlocks
    solution_blocks: SolutionBlocks
    final_demand_blocks: FinalDemandBlocks
    bottom_blocks: BottomBlocks
    weights: list[Array]
    regionalised: bool = False
    prior_blocks: PriorBlocks | None = None
    planted_solution: PlantedSolution | None = None

    @classmethod
    def from_configuration(
        cls,
        config: DisaggregationConfig,
        reader: ICIOReader,
        technical_coeffs_prior_df: Optional[pd.DataFrame] = None,
        final_demand_prior_df: Optional[pd.DataFrame] = None,
    ):
        """Create a DisaggregationProblem from a configuration and reader.

        Args:
            config: Configuration specifying the disaggregation structure
            reader: Original ICIO reader used for the disaggregation
            technical_coeffs_prior_df: Optional DataFrame containing prior information with columns:
                - For multi-country: [Country_row, Sector_row, Country_column, Sector_column, value]
                - For single-country: [Sector_row, Sector_column, value]
            final_demand_prior_df: Optional DataFrame containing final demand priors with columns:
                - For multi-country: [Country, Sector, value]
                - For single-country: [Sector, value]

        Returns:
            DisaggregationProblem instance containing both the problems to solve
            and the structure to hold the solution
        """
        # Get list of countries to keep separate
        countries_to_keep = config.get_countries_to_keep()

        # If we have a subset of countries, create a new reader with ROW aggregation
        if set(countries_to_keep) != set(reader.countries):
            if reader.data_path is None:
                raise ValueError(
                    "Cannot perform country aggregation on reader without data_path. "
                    "Please load reader from CSV file."
                )
            reader = ICIOReader.from_csv_selection(reader.data_path, countries_to_keep)

        disag_mapping = config.get_disagg_mapping()
        weight_dict = config.get_weight_dictionary()

        # Check if the disaggregation is regional
        regionalised = _check_regional(disag_mapping)

        # Setup the disaggregation blocks
        sectors_info = unfold_sectors_info(disag_mapping)
        blocks = DisaggregationBlocks.from_technical_coefficients(
            tech_coef=reader.technical_coefficients,
            sectors_info=sectors_info,
            output=reader.output_from_out,
        )

        # Create the solution blocks structure
        solution = SolutionBlocks.from_disaggregation_blocks(
            blocks,
            disag_mapping,
            weight_dict,
        )

        # Create the final demand blocks structure
        final_demand = FinalDemandBlocks.from_disaggregation_blocks(
            final_demand_table=reader.final_demand_table,
            output=blocks.output,
            disagg_mapping=disag_mapping,
            region_outputs=solution.output.groupby(level=0).sum(),
        )

        # Create the bottom blocks structure
        bottom = BottomBlocks.from_disaggregation_blocks(
            reader=reader,
            disagg_mapping=disag_mapping,
            weight_dict=weight_dict,
        )

        # Create weights list for each sector being disaggregated
        weights = []
        for sector in blocks.sectors:
            subsectors = disag_mapping[sector.sector_id]
            weights.append(np.array([weight_dict.get(subsector, 0) for subsector in subsectors]))

        # Handle prior information if provided
        prior_blocks = None
        if technical_coeffs_prior_df is not None:
            # Convert DataFrame to list of PriorInfo tuples
            prior_info = _convert_prior_df_to_info(technical_coeffs_prior_df)

            # Convert final demand prior if provided
            final_demand_prior = None
            if final_demand_prior_df is not None:
                is_multi_country = "Country" in final_demand_prior_df.columns
                final_demand_prior = []
                for _, row in final_demand_prior_df.iterrows():
                    sector = (row["Country"], row["Sector"]) if is_multi_country else row["Sector"]
                    final_demand_prior.append((sector, row["value"]))

            # Create prior blocks
            prior_blocks = PriorBlocks.from_disaggregation_blocks(
                blocks,
                disag_mapping,
                prior_info,
                final_demand_prior,
            )

        # Create individual problems for each sector
        problems = []
        for sector in blocks.sectors:
            subsectors = disag_mapping[sector.sector_id]
            problems.append(
                SectorDisaggregationProblem.from_disaggregation_block(
                    blocks,
                    sector.index,
                    weights,
                    sector,
                    sorted(list(subsectors)),
                    prior_blocks,
                )
            )

        # Create planted solution for testing
        planted_solution = PlantedSolution.from_disaggregation_blocks(
            blocks=blocks,
            disaggregation_dict=disag_mapping,
            weight_dict=weight_dict,
        )

        return cls(
            problems=problems,
            disaggregation_blocks=blocks,
            solution_blocks=solution,
            final_demand_blocks=final_demand,
            bottom_blocks=bottom,
            weights=weights,
            prior_blocks=prior_blocks,
            planted_solution=planted_solution,
            regionalised=regionalised,
        )

    def solve(
        self,
        lambda_sparse: float = 1.0,
        mu_prior: float = 10.0,
        use_planted_solution: bool = True,
    ) -> None:
        """Solve all sector problems and apply solutions to solution_blocks and final_demand_blocks.

        Args:
            lambda_sparse: Weight for L1 penalty on sparse terms (default: 1.0)
            mu_prior: Weight for L2 penalty on deviation from known terms (default: 10.0)
            use_planted_solution: Whether to use the planted solution as initial guess (default: True)
        """
        for n, problem in enumerate(self.problems, start=1):
            # If we have a planted solution and want to use it, get the appropriate vector
            initial_guess = None
            if use_planted_solution and self.planted_solution is not None:
                initial_guess = self.planted_solution.get_xn_vector(n)

            x_n = problem.solve(
                lambda_sparse=lambda_sparse,
                mu_prior=mu_prior,
                initial_guess=initial_guess,
                use_initial_guess=use_planted_solution,
            )
            self.solution_blocks.apply_xn(n, x_n)

            # Extract bn_vector from x_n (last k_n elements)
            k_n = len(problem.disaggregated_sectors)
            bn_vector = x_n[-k_n:]
            self.final_demand_blocks.apply_bn_vector(n, bn_vector)
