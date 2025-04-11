import numpy as np
import pytest

from disag_tools.disaggregation.planted_solution import PlantedSolution


@pytest.mark.parametrize("n", [1, 2])
def test_planted_large_equation(
    aggregated_blocks,
    n,
    disaggregated_blocks,
    usa_reader_blocks,
    usa_aggregated_reader,
):
    """Test that the planted solution satisfies the large equation M@X = Y.

    This test verifies that the planted solution (which is generated from the disaggregated data)
    satisfies the disaggregation equations for the real-world test case where sector A is
    disaggregated into A01 and A03.
    """
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(usa_reader_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = float(
                    usa_reader_blocks.output[subsector_id] / total_output  # type: ignore
                )

    # Create a planted solution using our new class
    planted_solution = PlantedSolution.from_disaggregation_blocks(
        blocks=aggregated_blocks,
        disaggregation_dict=disaggregation_dict,
        weight_dict=weight_dict,
    )

    # Get the x_n vector from our planted solution
    planted_xn = planted_solution.get_xn_vector(n)

    # Convert weight_dict into a list of numpy arrays
    # Each array contains weights for a sector's subsectors in order
    weight_arrays = []
    for sector_id in aggregated_blocks.to_disagg_sector_names:
        subsector_ids = disaggregation_dict[sector_id]
        weights = np.array([weight_dict[sid] for sid in subsector_ids])
        weight_arrays.append(weights)

    # Get the large matrix and target vector
    large_m = aggregated_blocks.get_large_m(n, weight_arrays)
    target_y = aggregated_blocks.get_y_vector(n, weight_arrays[n - 1])

    # Check that M@X = Y
    result = large_m @ planted_xn
    assert np.allclose(result, target_y, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2])
def test_m1_block_equation_planted(
    aggregated_blocks,
    n,
    disaggregated_blocks,
    usa_reader_blocks,
    usa_aggregated_reader,
):
    """Test the equation for the E block (flows from undisaggregated to disaggregated sectors)
    using the planted solution.

    The equation states that:
    E_{ij} = B_i w_j^n

    where:
    - E_{ij} is the flow from undisaggregated sector i to subsector j
    - B_i is the original flow from sector i to the aggregated sector
    - w_j^n is the weight for subsector j
    """
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(usa_reader_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = float(
                    usa_reader_blocks.output[subsector_id] / total_output  # type: ignore
                )

    # Create a planted solution using our new class
    planted_solution = PlantedSolution.from_disaggregation_blocks(
        blocks=aggregated_blocks,
        disaggregation_dict=disaggregation_dict,
        weight_dict=weight_dict,
    )

    # Get E vector from planted solution
    E = planted_solution.get_e_vector(n)
    weights = np.array(
        [
            weight_dict[sid]
            for sid in disaggregation_dict[aggregated_blocks.to_disagg_sector_names[n - 1]]
        ]
    )

    # Get M1 block and B vector
    M1 = aggregated_blocks.get_m1_block(n, weights)
    B = aggregated_blocks.get_B(n)

    # test M1E = B
    result = M1 @ E
    assert np.allclose(result, B, rtol=1e-2), "M1E ≠ B: Equation does not hold for planted solution"


@pytest.mark.parametrize("n", [1, 2])
def test_F_block_equation_planted(
    aggregated_blocks,
    n,
    disaggregated_blocks,
    usa_reader_blocks,
    usa_aggregated_reader,
):
    """Test the equation M₂^n F^n = C^n for the F block using the planted solution.

    According to the disaggregation plan:
    - F^n has shape (k_n × N_K) where:
        * k_n is the number of subsectors for sector n (2 in our case: A01, A03)
        * N_K is the number of undisaggregated sectors across all countries
    - M₂^n has shape (N_K × N_K*k_n)
    - C^n has shape (N_K × 1)

    The equation ensures that when we sum the flows from subsectors (F)
    using the constraint matrix (M₂), we get the original aggregated flows (C).
    """
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(usa_reader_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = float(
                    usa_reader_blocks.output[subsector_id] / total_output  # type: ignore
                )

    # Create a planted solution using our new class
    planted_solution = PlantedSolution.from_disaggregation_blocks(
        blocks=aggregated_blocks,
        disaggregation_dict=disaggregation_dict,
        weight_dict=weight_dict,
    )

    # Get F vector from planted solution
    F = planted_solution.get_f_vector(n)
    M2 = aggregated_blocks.get_m2_block(n)
    C = aggregated_blocks.get_C(n)

    # test M2F = C
    result = M2 @ F
    assert np.allclose(result, C, rtol=1e-2), "M2F ≠ C: Equation does not hold for planted solution"


@pytest.mark.parametrize("n, l", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_Gnl_block_equation_planted(
    aggregated_blocks,
    n,
    l,
    disaggregated_blocks,
    usa_reader_blocks,
    usa_aggregated_reader,
):
    """Test the equation M₃^n G^n = D^n for the G block using the planted solution.

    The equation ensures that when we apply the weight constraints (M₃)
    to the subsector flows (G), we get the original aggregated flows (D).
    """
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(usa_reader_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = float(
                    usa_reader_blocks.output[subsector_id] / total_output  # type: ignore
                )

    # Create a planted solution using our new class
    planted_solution = PlantedSolution.from_disaggregation_blocks(
        blocks=aggregated_blocks,
        disaggregation_dict=disaggregation_dict,
        weight_dict=weight_dict,
    )

    # Get G vector from planted solution
    g_nl = planted_solution.get_gnl_vector(n, l)
    weights_l = np.array(
        [
            weight_dict[sid]
            for sid in disaggregation_dict[aggregated_blocks.to_disagg_sector_names[l - 1]]
        ]
    )
    M3 = aggregated_blocks.get_m3_nl_block(n, weights_l)
    d_nl = aggregated_blocks.get_D_nl(n, l)

    # test M3G = D
    result = M3 @ g_nl
    assert np.allclose(
        result, d_nl, rtol=1e-2
    ), f"M3G ≠ D: Equation does not hold for planted solution (n={n}, l={l})"


@pytest.mark.parametrize("n", [1, 2])
def test_final_demand_block_equation_planted(
    aggregated_blocks,
    n,
    disaggregated_blocks,
    usa_reader_blocks,
    usa_aggregated_reader,
):
    """Test the final demand block equation M_5F + M_4G + b = w using planted solution.

    Args:
        aggregated_blocks: Original aggregated blocks
        n: Sector index (1-based)
        disaggregated_blocks: Disaggregated blocks
        usa_reader_blocks: Reader blocks for the USA
        usa_aggregated_reader: Reader for the USA aggregated data
    """
    base_mapping = {"A": ["A01", "A03"]}
    countries = list(aggregated_blocks.reordered_matrix.index.get_level_values(0).unique())

    # Create the full mapping for all countries
    disaggregation_dict = {}
    weight_dict = {}
    for country in countries:
        for sector, subsectors in base_mapping.items():
            sector_id = (country, sector)
            subsector_ids = [(country, s) for s in subsectors]
            disaggregation_dict[sector_id] = subsector_ids

            # Compute weights from the output values
            total_output = sum(usa_reader_blocks.output[sid] for sid in subsector_ids)
            for subsector_id in subsector_ids:
                weight_dict[subsector_id] = float(
                    usa_reader_blocks.output[subsector_id] / total_output  # type: ignore
                )

    # Create a planted solution using our new class
    planted_solution = PlantedSolution.from_disaggregation_blocks(
        blocks=aggregated_blocks,
        disaggregation_dict=disaggregation_dict,
        weight_dict=weight_dict,
    )

    # Get the weights for this sector's subsectors
    w = np.array(
        [
            weight_dict[sid]
            for sid in disaggregation_dict[aggregated_blocks.to_disagg_sector_names[n - 1]]
        ]
    )

    # Get F and G vectors
    F = planted_solution.get_f_vector(n)
    G = planted_solution.get_gn_vector(n)

    # Get M4 and M5 matrices
    # Get all relative weights for all sectors
    relative_weights = []
    for l in range(1, aggregated_blocks.K + 1):
        sector_id = aggregated_blocks.to_disagg_sector_names[l - 1]
        weights = np.array([weight_dict[sid] for sid in disaggregation_dict[sector_id]])
        relative_weights.append(weights)
    M4 = aggregated_blocks.get_m4_block(n, relative_weights)
    M5 = aggregated_blocks.get_m5_block(n)

    # Get b_n vector
    b_n = planted_solution.get_bn_vector(n)

    # Check the equation M_5F + M_4G + b = w
    result = M5 @ F + M4 @ G + b_n
    np.testing.assert_allclose(result, w, rtol=1e-2)
