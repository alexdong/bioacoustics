# taxonomy.py
"""Functions for handling taxonomy data and calculating distances."""

import time
from itertools import product
from pathlib import Path
from typing import Any

import config  # Import config variables
import numpy as np
import pandas as pd

# Type alias for clarity
Species = str
Genus = str
Family = str
Order = str
TaxonRank = str
Lineage = list[TaxonRank]
TaxonomyTree = dict[TaxonRank, dict[str, Any]] # Simple nested dict representation
DistanceMatrix = np.ndarray

# --- Top Level Function ---
def load_and_compute_distance_matrix(
    taxonomy_path: Path, species_list: list[Species],
) -> tuple[dict[Species, Lineage], DistanceMatrix]:
    """Loads taxonomy, builds tree, and computes pairwise distance matrix."""
    print("[TAXONOMY] Loading taxonomy data...")
    assert taxonomy_path.is_file(), f"Taxonomy file not found: {taxonomy_path}"
    taxonomy_df = pd.read_csv(taxonomy_path)
    # TODO: Adapt column names based on the actual taxonomy.csv file
    # Assuming columns like: primary_label (species), genus, family, order
    required_cols = ["primary_label", "genus", "family", "order"] # TODO: Adjust!
    assert all(col in taxonomy_df.columns for col in required_cols), \
        f"Missing required columns in taxonomy file: {required_cols}"

    print("[TAXONOMY] Building internal lineage representation...")
    lineage_map = _build_lineage_map(taxonomy_df, species_list)

    print("[TAXONOMY] Precomputing pairwise distance matrix...")
    start_time = time.time()
    distance_matrix = _compute_pairwise_distances(lineage_map, species_list)
    end_time = time.time()
    print(f"[TAXONOMY] Distance matrix computation took {end_time - start_time:.2f}s")

    return lineage_map, distance_matrix

# --- Helper Functions ---
def _build_lineage_map(df: pd.DataFrame, species_list: list[Species]) -> dict[Species, Lineage]:
    """Creates a map from species name to its taxonomic lineage."""
    lineage_map: dict[Species, Lineage] = {}
    species_set = set(species_list)
    # TODO: Adjust column names as needed
    df_filtered = df[df['primary_label'].isin(species_set)].drop_duplicates(subset=['primary_label'])

    for _, row in df_filtered.iterrows():
        species = row['primary_label']
        # Lineage: Order -> Family -> Genus -> Species (higher ranks first)
        # TODO: Adapt ranks and column names if necessary
        lineage = [row['order'], row['family'], row['genus'], species]
        lineage_map[species] = lineage

    # Assert all species in the list were found in the taxonomy file
    assert len(lineage_map) == len(species_list), \
        f"Mismatch: {len(species_list)} species required, {len(lineage_map)} found in taxonomy."
    return lineage_map

def _get_lca_depth(lineage1: Lineage, lineage2: Lineage) -> int:
    """Finds the depth of the Lowest Common Ancestor (LCA). Depth 0 is root."""
    lca_depth = 0
    for rank1, rank2 in zip(lineage1, lineage2):
        if rank1 == rank2:
            lca_depth += 1
        else:
            break
    return lca_depth

def _compute_taxonomic_distance(lineage1: Lineage, lineage2: Lineage) -> int:
    """Calculates distance based on steps to LCA."""
    if lineage1 == lineage2:
        return 0
    lca_depth = _get_lca_depth(lineage1, lineage2)
    # Depth of a node is number of steps from root. LCA depth is steps from root to LCA.
    # Steps from species to root = len(lineage)
    # Steps from species to LCA = len(lineage) - lca_depth
    dist = (len(lineage1) - lca_depth) + (len(lineage2) - lca_depth)
    return dist

def _compute_pairwise_distances(
    lineage_map: dict[Species, Lineage], species_list: list[Species],
) -> DistanceMatrix:
    """Computes the N x N distance matrix."""
    num_species = len(species_list)
    distance_matrix = np.zeros((num_species, num_species), dtype=np.int32)
    {name: i for i, name in enumerate(species_list)}

    # Use itertools.product for potentially cleaner iteration (optional)
    for i, j in product(range(num_species), range(num_species)):
        if i == j:
            continue
        species_i = species_list[i]
        species_j = species_list[j]
        lineage_i = lineage_map[species_i]
        lineage_j = lineage_map[species_j]
        dist = _compute_taxonomic_distance(lineage_i, lineage_j)
        distance_matrix[i, j] = dist

    # Basic check
    assert np.all(distance_matrix >= 0), "Distances should be non-negative"
    assert np.all(distance_matrix == distance_matrix.T), "Distance matrix should be symmetric"
    assert np.all(np.diag(distance_matrix) == 0), "Distance to self should be 0"

    return distance_matrix

# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running taxonomy.py demonstration ---")

    # Create dummy data for testing
    dummy_taxonomy_data = {
        'primary_label': ['sp_a1', 'sp_a2', 'sp_b1', 'sp_c1'],
        'genus':         ['gen_a', 'gen_a', 'gen_b', 'gen_c'],
        'family':        ['fam_a', 'fam_a', 'fam_b', 'fam_c'],
        'order':         ['ord_x', 'ord_x', 'ord_x', 'ord_y'],
    }
    dummy_df = pd.DataFrame(dummy_taxonomy_data)
    dummy_path = config.TEMP_DIR / "dummy_taxonomy.csv"
    dummy_df.to_csv(dummy_path, index=False)
    print(f"[DEMO] Created dummy taxonomy file: {dummy_path}")

    dummy_species_list = ['sp_a1', 'sp_a2', 'sp_b1', 'sp_c1']

    try:
        lineage_map_demo, dist_matrix_demo = load_and_compute_distance_matrix(
            dummy_path, dummy_species_list,
        )

        print("\n[DEMO] Lineage Map:")
        for species, lineage in lineage_map_demo.items():
            print(f"  {species}: {lineage}")

        print("\n[DEMO] Distance Matrix:")
        print(dist_matrix_demo)

        # Example distance check
        idx_a1 = dummy_species_list.index('sp_a1')
        idx_a2 = dummy_species_list.index('sp_a2')
        idx_b1 = dummy_species_list.index('sp_b1')
        idx_c1 = dummy_species_list.index('sp_c1')

        print(f"\n[DEMO] Dist(sp_a1, sp_a2) = {dist_matrix_demo[idx_a1, idx_a2]} (Expected: 2)")
        print(f"[DEMO] Dist(sp_a1, sp_b1) = {dist_matrix_demo[idx_a1, idx_b1]} (Expected: 4)")
        print(f"[DEMO] Dist(sp_a1, sp_c1) = {dist_matrix_demo[idx_a1, idx_c1]} (Expected: 6)") # Different order

    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")

    # NOTE: As per convention, dummy file is NOT cleaned up from /tmp
    print("--- End taxonomy.py demonstration ---")
