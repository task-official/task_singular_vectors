"""Explicit historical names; every executable kernel lives under src.methods."""
from src.compat.kernels import (
    compute_and_sum_svd_mem_reduction,
    compute_component_selection_tsv_merge,
    compute_adaptive_tsv_merge,
    compute_adaptive_tsv_subspace,
    compute_and_sum_svd_mem_reduction_2,
    compute_and_sum_svd_mem_reduction_dummy,
)

__all__ = [
    "compute_and_sum_svd_mem_reduction", "compute_component_selection_tsv_merge",
    "compute_adaptive_tsv_merge", "compute_adaptive_tsv_subspace",
    "compute_and_sum_svd_mem_reduction_2", "compute_and_sum_svd_mem_reduction_dummy",
]
