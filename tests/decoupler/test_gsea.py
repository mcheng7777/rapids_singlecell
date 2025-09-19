from __future__ import annotations

import gseapy as gp
import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

import rapids_singlecell.decoupler_gpu as dc
from rapids_singlecell.decoupler_gpu._method_gsea import (
    _func_gsea as _func_gsea_optimized,
    launch_batch_gsea_kernel,
)


# =============================================================================
# PHASE 1: UNIT TESTS
# =============================================================================
def test_ridx():
    idx_a = dc._method_gsea._ridx(times=5, nvar=10, seed=42)
    assert (~(cp.diff(idx_a) == 1).all(axis=1)).all()
    idx_b = dc._method_gsea._ridx(times=5, nvar=10, seed=2)
    assert (~(cp.diff(idx_b) == 1).all(axis=1)).all()
    assert (~(idx_a == idx_b).all(axis=1)).all()

# def test_es_kernel():
#     """Test ES kernel functionality."""
#     n_features = 3
#     n_sources = 1
    
#     # Test data matching original decoupler test cases
#     row = cp.array([0.0, 2.0, 0.0], dtype=cp.float32)
#     rnks = cp.array([0, 1, 2], dtype=cp.int32)
#     set_msk_batch = cp.array([[False, True, False]], dtype=cp.bool_)
#     dec_batch = cp.array([0.1], dtype=cp.float32)
    
#     es_results, es_indices = launch_es_kernel(
#         row, rnks, set_msk_batch, dec_batch, n_sources, n_features
#     )
    
#     assert es_results.shape == (n_sources,)
#     assert es_indices.shape == (n_sources,)
#     assert cp.all(cp.isfinite(es_results))
#     assert cp.isclose(es_results[0], 0.9, atol=1e-6)
#     assert es_indices[0] == 1

# def test_nes_kernel():
#     """Test NES kernel functionality."""
#     n_features = 3
#     n_sources = 1
#     n_permutations = 6
    
#     ridx = cp.array(
#         [
#             [0, 1, 2],
#             [0, 2, 1],
#             [1, 2, 0],
#             [1, 0, 2],
#             [2, 0, 1],
#             [2, 1, 0],
#         ],
#         dtype=cp.int32
#     )
#     row = cp.array([0.0, 2.0, 0.0], dtype=cp.float32)
#     rnks = cp.array([0, 1, 2], dtype=cp.int32)
#     set_msk_batch = cp.array([[False, True, False]], dtype=cp.bool_)
#     dec_batch = cp.array([0.1], dtype=cp.float32)
#     es_batch = cp.array([0.9], dtype=cp.float32)
    
#     nes_results, pv_results = launch_nes_kernel(
#         ridx, row, rnks, set_msk_batch, dec_batch, es_batch,
#         n_sources, n_features, n_permutations
#     )
    
#     assert nes_results.shape == (n_sources,)
#     assert pv_results.shape == (n_sources,)
#     assert cp.all(cp.isfinite(nes_results))
#     assert cp.all((pv_results >= 0) & (pv_results <= 1))


def test_batch_gsea_kernel(
    mat,
    idxmat,
):
    """Test batch GSEA kernel functionality."""
    X, obs, var = mat
    cnct, starts, offsets = idxmat
    
    # Convert to CuPy
    X_cp = cp.array(X, dtype=cp.float32)
    cnct_cp = cp.array(cnct, dtype=cp.int32)
    starts_cp = cp.array(starts, dtype=cp.int32)
    offsets_cp = cp.array(offsets, dtype=cp.int32)
    
    # Create test data for batch kernel
    n_obs, n_features = X_cp.shape
    n_sources = starts_cp.size
    times = 10
    
    # Pre-compute feature sets and masks
    set_msk_batch = cp.zeros((n_sources, n_features), dtype=cp.bool_)
    dec_batch = cp.zeros(n_sources, dtype=cp.float32)
    
    for j in range(n_sources):
        fset = dc._method_gsea._getset(cnct_cp, starts_cp, offsets_cp, j)
        set_msk_batch[j, fset] = True
        dec_batch[j] = 1.0 / (n_features - fset.size)
    
    # Generate permutation indices
    ridx = dc._method_gsea._ridx(times=times, nvar=n_features, seed=42)
    rnks = cp.argsort(-X_cp, axis=1).astype(cp.int32)
    
    # Test batch kernel
    
    es, nes, pv = launch_batch_gsea_kernel(
        X_cp, set_msk_batch, dec_batch, ridx, rnks,
        n_obs, n_sources, n_features, times
    )
    assert es.size == offsets.size
    assert nes.size == offsets.size
    assert pv.size == offsets.size


def test_func_gsea(
    mat,
    net,
    idxmat,
):
    times = 10  # Reduced for faster testing
    seed = 42
    X, obs, var = mat
    gene_sets = net.groupby("source")["target"].apply(lambda x: list(x)).to_dict()
    cnct, starts, offsets = idxmat
    row = cp.array(X[0, :])  # Convert to CuPy
    cnct = cp.array(cnct)    # Convert to CuPy
    starts = cp.array(starts)  # Convert to CuPy
    offsets = cp.array(offsets)  # Convert to CuPy
    res = gp.prerank(
        rnk=pd.DataFrame(X, index=obs, columns=var).T,
        gene_sets=gene_sets,
        permutation_num=times,
        permutation_type="gene_set",
        outdir=None,
        min_size=0,
        threads=4,
        seed=seed,
    ).res2d
    gp_es = res.pivot(index="Name", columns="Term", values="NES").astype(float)
    dc_es, dc_pv = dc._method_gsea._func_gsea(
        mat=cp.array(X),
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        times=times,
        seed=seed,
    )
    assert (gp_es - dc_es).abs().values.max() < 0.10



