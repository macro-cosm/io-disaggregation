"""Command line interface for IO table disaggregation."""

import logging
from pathlib import Path

import click
import pandas as pd
import yaml

from disag_tools.assembler.assembler import AssembledData
from disag_tools.configurations.config import DisaggregationConfig
from disag_tools.disaggregation.problem import DisaggregationProblem
from disag_tools.readers.icio_reader import ICIOReader

logger = logging.getLogger(__name__)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--prior-info",
    type=click.Path(exists=True, path_type=Path),
    help="Path to CSV file containing prior information for technical coefficients",
)
@click.option(
    "--final-demand-prior",
    type=click.Path(exists=True, path_type=Path),
    help="Path to CSV file containing prior information for final demand",
)
@click.option(
    "--lambda-sparse",
    type=float,
    default=1.0,
    help="Weight for L1 penalty on sparse terms (default: 1.0)",
)
@click.option(
    "--mu-prior",
    type=float,
    default=10.0,
    help="Weight for L2 penalty on deviation from known terms (default: 10.0)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Set the logging level",
)
def disaggregate(
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    prior_info: Path | None,
    final_demand_prior: Path | None,
    lambda_sparse: float,
    mu_prior: float,
    log_level: str,
) -> None:
    """Disaggregate an IO table using the specified configuration.

    Args:
        config_path: Path to YAML configuration file
        input_path: Path to input CSV file containing the IO table
        output_dir: Directory to write output files to
        prior_info: Optional path to CSV with prior information for technical coefficients
        final_demand_prior: Optional path to CSV with prior information for final demand
        lambda_sparse: Weight for L1 penalty on sparse terms
        mu_prior: Weight for L2 penalty on deviation from known terms
        log_level: Logging level to use
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = DisaggregationConfig.model_validate(config_dict)

    # Load IO table
    logger.info(f"Loading IO table from {input_path}")
    reader = ICIOReader.from_csv(input_path)

    # Get list of countries to keep separate
    countries_to_keep = config.get_countries_to_keep()
    if set(countries_to_keep) != set(reader.countries):
        logger.info(f"Creating reader with selected countries: {countries_to_keep}")
        reader = ICIOReader.from_csv_selection(input_path, countries_to_keep)

    # Load prior information if provided
    prior_df = None
    if prior_info is not None:
        logger.info(f"Loading prior information from {prior_info}")
        prior_df = pd.read_csv(prior_info)

    final_demand_prior_df = None
    if final_demand_prior is not None:
        logger.info(f"Loading final demand prior from {final_demand_prior}")
        final_demand_prior_df = pd.read_csv(final_demand_prior)

    # Create and solve the disaggregation problem
    logger.info("Creating disaggregation problem")
    problem = DisaggregationProblem.from_configuration(
        config=config,
        reader=reader,
        technical_coeffs_prior_df=prior_df,
        final_demand_prior_df=final_demand_prior_df,
    )

    logger.info("Solving disaggregation problem")
    problem.solve(lambda_sparse=lambda_sparse, mu_prior=mu_prior)

    # Save results
    logger.info(f"Saving results to {output_dir}")

    # Create assembled data from solution
    assembled = AssembledData.from_solution(problem, reader)

    # Save the assembled table
    output_path = output_dir / "disaggregated_table.csv"
    assembled.data.to_csv(output_path)
    logger.info(f"Saved disaggregated table to {output_path}")

    logger.info("Done!")


if __name__ == "__main__":
    disaggregate()
