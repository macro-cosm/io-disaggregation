"""Pydantic models for disaggregation configuration."""

from typing import Annotated, Dict, List

from pydantic import BaseModel, Field, NonNegativeFloat, model_validator


class SubsectorConfig(BaseModel):
    """Configuration for a single sub-sector."""

    name: str = Field(..., description="Human-readable name for the sub-sector")
    relative_output_weight: Annotated[float, Field(..., ge=0, le=1)] = Field(
        ..., description="Relative output weight of this sub-sector"
    )


class RegionConfig(BaseModel):
    """Configuration for a single region within a country."""

    name: str = Field(..., description="Human-readable name for the region")
    sector_weights: Dict[str, float] = Field(
        ..., description="Mapping of sector codes to their relative output weights in this region"
    )


class SectorConfig(BaseModel):
    """Configuration for disaggregating a sector."""

    subsectors: Dict[str, SubsectorConfig] = Field(
        ..., description="Mapping of sub-sector codes to their configurations"
    )

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "SectorConfig":
        """Ensure subsector weights sum to 1."""
        total = sum(sub.relative_output_weight for sub in self.subsectors.values())
        if not abs(total - 1.0) < 1e-10:  # Using small epsilon for float comparison
            raise ValueError(
                f"Subsector weights must sum to 1, got {total} for sector {list(self.subsectors.keys())[0]}"
            )
        return self


class CountryConfig(BaseModel):
    """Configuration for disaggregating a country."""

    regions: Dict[str, RegionConfig] = Field(
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
    """Top-level configuration for disaggregation."""

    countries: Dict[str, CountryConfig] | None = Field(
        None, description="Optional mapping of country codes to their disaggregation configs"
    )
    sectors: Dict[str, SectorConfig] | None = Field(
        None, description="Optional mapping of sector codes to their disaggregation configs"
    )

    @model_validator(mode="after")
    def validate_not_empty(self) -> "DisaggregationConfig":
        """Ensure at least one type of disaggregation is specified."""
        if not self.countries and not self.sectors:
            raise ValueError("Must specify at least one country or sector to disaggregate")
        return self

    def get_final_size(
        self, original_countries: List[str], original_sectors: List[str]
    ) -> tuple[int, int]:
        """
        Calculate the final number of rows/columns after disaggregation.

        Args:
            original_countries: List of country codes in the original matrix
            original_sectors: List of sector codes in the original matrix

        Returns:
            Tuple of (rows, cols) in the final matrix
        """
        # Start with original size
        n_countries = len(original_countries)
        n_sectors = len(original_sectors)

        # Add new regions (each region gets all sectors)
        if self.countries:
            for country, config in self.countries.items():
                if country in original_countries:
                    # Subtract 1 for original country, add number of regions
                    n_countries += len(config.regions) - 1

        # Add new subsectors (each subsector appears in all countries/regions)
        if self.sectors:
            for sector, config in self.sectors.items():
                if sector in original_sectors:
                    # Subtract 1 for original sector, add number of subsectors
                    n_sectors += len(config.subsectors) - 1

        # Final size is product of countries and sectors
        final_size = n_countries * n_sectors
        return final_size, final_size

    def get_disagg_mapping(self) -> Dict[tuple[str, str], List[tuple[str, str]]]:
        """
        Get mapping from original (country, sector) pairs to disaggregated pairs.

        Returns:
            Dict mapping (country, sector) to list of disaggregated (country, sector) pairs
        """
        mapping: Dict[tuple[str, str], List[tuple[str, str]]] = {}

        # Helper to process a single country-sector pair
        def process_pair(country: str, sector: str) -> List[tuple[str, str]]:
            result = [(country, sector)]  # Start with original pair

            # Apply country disaggregation if applicable
            if self.countries and country in self.countries:
                # Only map if the sector has weights defined for this region
                result = [
                    (region_code, sector)
                    for region_code, region in self.countries[country].regions.items()
                    if sector in region.sector_weights
                ]

            # Apply sector disaggregation if applicable
            if self.sectors and sector in self.sectors:
                # If country was disaggregated, apply to all regions
                # Otherwise, apply to original country
                current_countries = {pair[0] for pair in result}
                result = [
                    (country, subsector_code)
                    for country in current_countries
                    for subsector_code in self.sectors[sector].subsectors.keys()
                ]

            return result

        # If we have country disaggregation
        if self.countries:
            for country, country_config in self.countries.items():
                # Get all sectors that have weights defined in any region
                all_sectors = set()
                for region in country_config.regions.values():
                    all_sectors.update(region.sector_weights.keys())

                # Map each sector
                for sector in all_sectors:
                    orig_pair = (country, sector)
                    mapping[orig_pair] = process_pair(country, sector)

        # If we have sector disaggregation
        if self.sectors:
            for sector in self.sectors:
                # For each country that might be disaggregated
                if self.countries:
                    for country in self.countries:
                        # Skip if already processed in country disaggregation
                        orig_pair = (country, sector)
                        if orig_pair not in mapping:
                            mapping[orig_pair] = process_pair(country, sector)
                else:
                    # Just use empty string as placeholder for country
                    orig_pair = ("", sector)
                    mapping[orig_pair] = process_pair("", sector)

        return mapping
