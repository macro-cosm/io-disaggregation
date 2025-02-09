"""Pydantic models for disaggregation configuration."""

from itertools import product
from typing import Tuple

from pydantic import BaseModel, Field, NonNegativeFloat, field_validator, model_validator


class SubsectorConfig(BaseModel):
    """Configuration for a subsector in sector disaggregation."""

    name: str
    relative_output_weights: dict[str, float]  # Maps country code to relative output weight

    @field_validator("relative_output_weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        """Validate that weights are between 0 and 1."""
        for country, weight in weights.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for country {country} must be between 0 and 1")
        return weights


class RegionConfig(BaseModel):
    """Configuration for a single region within a country."""

    name: str = Field(..., description="Human-readable name for the region")
    sector_weights: dict[str, float] = Field(
        ..., description="Mapping of sector codes to their relative output weights in this region"
    )


class SectorConfig(BaseModel):
    """Configuration for sector disaggregation."""

    subsectors: dict[str, SubsectorConfig]

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "SectorConfig":
        """Validate that weights sum to 1 for each country."""
        # Get all countries from all subsectors
        all_countries = {
            country
            for subsector in self.subsectors.values()
            for country in subsector.relative_output_weights.keys()
        }

        # Check that all subsectors have weights for all countries
        for subsector_code, subsector in self.subsectors.items():
            missing_countries = all_countries - set(subsector.relative_output_weights.keys())
            if missing_countries:
                raise ValueError(
                    f"Subsector {subsector_code} is missing weights for countries: {missing_countries}"
                )

        # Check that weights sum to 1 for each country
        for country in all_countries:
            total = sum(
                subsector.relative_output_weights[country] for subsector in self.subsectors.values()
            )
            if not abs(total - 1.0) < 1e-10:  # Use small epsilon for float comparison
                raise ValueError(
                    f"Weights for country {country} sum to {total}, but should sum to 1"
                )

        return self


class CountryConfig(BaseModel):
    """Configuration for disaggregating a country."""

    regions: dict[str, RegionConfig] = Field(
        ..., description="Mapping of region codes to their configurations"
    )

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "CountryConfig":
        """Ensure region weights sum to 1 for each sector."""
        # Get all unique sectors across all regions
        all_sectors = set()
        for region in self.regions.values():
            all_sectors.update(region.sector_weights.keys())

        # Check each sector's weights sum to 1
        for sector in all_sectors:
            total = sum(region.sector_weights.get(sector, 0.0) for region in self.regions.values())
            if not abs(total - 1.0) < 1e-10:  # Using small epsilon for float comparison
                raise ValueError(f"Region weights for sector {sector} must sum to 1, got {total}")
        return self


class DisaggregationConfig(BaseModel):
    """Configuration for both sector and country disaggregation."""

    countries: dict[str, CountryConfig] | None = None
    sectors: dict[str, SectorConfig] | None = None

    @model_validator(mode="after")
    def validate_at_least_one_disagg(self) -> "DisaggregationConfig":
        """Validate that at least one type of disaggregation is specified."""
        if not self.countries and not self.sectors:
            raise ValueError("Must specify at least one country or sector to disaggregate")
        return self

    @staticmethod
    def _get_combinatorial_mapping(
        sector_mapping: dict[tuple[str, str], set[tuple[str, str]]],
        country_mapping: dict[tuple[str, str], set[tuple[str, str]]],
    ) -> dict[tuple[str, str], set[tuple[str, str]]]:
        """Combine sector and country mappings to create final mapping.

        This function handles the combinatorics of combining sector and country mappings.
        For any (country, sector) pair that appears in both mappings, it creates all
        possible combinations of the disaggregated pairs.

        Args:
            sector_mapping: Mapping from original pairs to disaggregated pairs for sectors
            country_mapping: Mapping from original pairs to disaggregated pairs for countries

        Returns:
            Combined mapping incorporating both sector and country disaggregation
        """
        # If either mapping is empty, return the other one
        if not sector_mapping:
            return country_mapping
        if not country_mapping:
            return sector_mapping

        # Start with all pairs from both mappings
        result = {}
        all_pairs = set(sector_mapping.keys()) | set(country_mapping.keys())

        for orig_pair in all_pairs:
            if orig_pair in sector_mapping and orig_pair in country_mapping:
                # Need to create combinations
                # Get all regions from country mapping
                regions = {pair[0] for pair in country_mapping[orig_pair]}
                # Get all subsectors from sector mapping
                subsectors = {pair[1] for pair in sector_mapping[orig_pair]}
                # Create all combinations
                result[orig_pair] = {
                    (region, subsector) for region, subsector in product(regions, subsectors)
                }
            elif orig_pair in sector_mapping:
                result[orig_pair] = sector_mapping[orig_pair]
            else:
                result[orig_pair] = country_mapping[orig_pair]

        return result

    def get_disagg_mapping(self) -> dict[tuple[str, str], set[tuple[str, str]]]:
        """Get mapping from original (country, sector) pairs to disaggregated pairs.

        The mapping depends on whether we have country disaggregation, sector disaggregation, or both.
        The mapping is used to determine how each (country, sector) pair should be split into new pairs.

        Examples:
            Sector-only disaggregation:
                Input config:
                    sectors:
                        MFG:
                            subsectors:
                                MFG1:
                                    relative_output_weights:
                                        USA: 0.6
                                        CHN: 0.7
                                        ROW: 0.5
                                MFG2:
                                    relative_output_weights:
                                        USA: 0.4
                                        CHN: 0.3
                                        ROW: 0.5

                Output mapping:
                    ("", "MFG"): {("", "MFG1"), ("", "MFG2")}  # For empty string case
                    ("USA", "MFG"): {("USA", "MFG1"), ("USA", "MFG2")}  # For country-specific case
                    ("CHN", "MFG"): {("CHN", "MFG1"), ("CHN", "MFG2")}
                    ("ROW", "MFG"): {("ROW", "MFG1"), ("ROW", "MFG2")}

            Country-only disaggregation:
                Input config:
                    countries:
                        CHN:
                            regions:
                                CHN1:
                                    relative_output_weights:
                                        MFG: 0.4
                                        SRV: 0.5
                                CHN2:
                                    relative_output_weights:
                                        MFG: 0.6
                                        SRV: 0.5

                Output mapping:
                    ("CHN", "MFG"): {("CHN1", "MFG"), ("CHN2", "MFG")}
                    ("CHN", "SRV"): {("CHN1", "SRV"), ("CHN2", "SRV")}

            Combined disaggregation:
                Input config:
                    sectors:
                        MFG:
                            subsectors:
                                MFG1:
                                    relative_output_weights:
                                        CHN: 0.7
                                MFG2:
                                    relative_output_weights:
                                        CHN: 0.3
                    countries:
                        CHN:
                            regions:
                                CHN1:
                                    relative_output_weights:
                                        MFG: 0.4
                                CHN2:
                                    relative_output_weights:
                                        MFG: 0.6

                Output mapping:
                    ("CHN", "MFG"): {
                        ("CHN1", "MFG1"), ("CHN1", "MFG2"),
                        ("CHN2", "MFG1"), ("CHN2", "MFG2")
                    }

        Returns:
            A dictionary mapping original (country, sector) pairs to sets of disaggregated pairs.
        """
        sector_mapping: dict[tuple[str, str], set[tuple[str, str]]] = {}
        country_mapping: dict[tuple[str, str], set[tuple[str, str]]] = {}

        # Handle sector disaggregation
        if self.sectors:
            for sector, config in self.sectors.items():
                # Get all countries that have weights defined for this sector
                countries = set()
                for subsector_config in config.subsectors.values():
                    countries.update(subsector_config.relative_output_weights.keys())

                # Create mappings for each country that has weights defined
                for country in countries:
                    orig_pair = (country, sector)
                    sector_mapping[orig_pair] = {
                        (country, subsector) for subsector in config.subsectors
                    }

        # Handle country disaggregation
        if self.countries:
            for country, config in self.countries.items():
                # Get all sectors that have weights defined for this country
                sectors = set()
                for region_config in config.regions.values():
                    sectors.update(region_config.sector_weights.keys())

                for sector in sectors:
                    orig_pair = (country, sector)
                    # Just map to regions
                    country_mapping[orig_pair] = {(region, sector) for region in config.regions}

        # Combine the mappings
        return self._get_combinatorial_mapping(sector_mapping, country_mapping)

    def get_final_size(
        self, original_countries: list[str], original_sectors: list[str]
    ) -> tuple[int, int]:
        """
        Compute the final size of the disaggregated matrix.

        Args:
            original_countries: List of countries before disaggregation
            original_sectors: List of sectors before disaggregation

        Returns:
            Tuple of (rows, cols) for final matrix size
        """
        # Start with original size
        n_countries = len(original_countries)
        n_sectors = len(original_sectors)

        # Add country disaggregation
        if self.countries:
            # Subtract original countries and add new regions
            n_countries = (
                n_countries
                - len(self.countries)
                + sum(len(c.regions) for c in self.countries.values())
            )

        # Add sector disaggregation
        if self.sectors:
            # Subtract original sectors and add new subsectors
            n_sectors = (
                n_sectors
                - len(self.sectors)
                + sum(len(s.subsectors) for s in self.sectors.values())
            )

        # Final size is countries × sectors
        final_size = n_countries * n_sectors
        return final_size, final_size
