from typing import TypeAlias

SectorId: TypeAlias = str | tuple[str, str]


def _check_regional(disagg_mapping: dict[SectorId, list[SectorId]]) -> bool:
    is_regional = False
    for key, value in disagg_mapping.items():
        if isinstance(key, tuple):
            first_country = key[0]
            # Only check if any subsector has a different country code
            regionalised = any(
                isinstance(subsector, tuple) and subsector[0] != first_country
                for subsector in value
            )
            is_regional = is_regional or regionalised
    return is_regional
