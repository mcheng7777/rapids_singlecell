from __future__ import annotations

import gseapy as gp
import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

import rapids_singlecell.decoupler_gpu as dc

def test_func_gsea(
    mat,
    net,
    idxmat,
):
    times = 2  # Reduced for faster testing
    seed = 42
    verbose = True
    X, obs, var = mat
    gene_sets = net.groupby("source")["target"].apply(lambda x: list(x)).to_dict()
    cnct, starts, offsets = idxmat
    
    # Convert to CuPy arrays for GPU implementation
    cnct = cp.array(cnct)
    starts = cp.array(starts)
    offsets = cp.array(offsets)
    X_gpu = cp.array(X, dtype=cp.float32)
    
    print(f"X shape: {X.shape}")
    # Run gseapy with original NumPy arrays
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
    gp_es = res.pivot(index="Name", columns="Term", values="NES").astype(np.float32)
    print(f"GSEApy NES values:\\n{gp_es}")
    # Run GPU implementation
    dc_es, dc_pv = dc._method_gsea._func_gsea(
        mat=X_gpu,
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        times=times,
        seed=seed,
        verbose=verbose,
    )
    print(f"GPU NES values:\\n{dc_es}")
    print(f"GPU P-values:\\n{dc_pv}")
    print(f"difference:\\n{gp_es - dc_es}")
    # Ensure arrays have the same shape
    assert gp_es.shape == dc_es.shape, f"Shape mismatch: gseapy {gp_es.shape} vs GPU {dc_es.shape}"
    # Test that p-values are in valid range [0, 1]
    assert (dc_pv >= 0).all() and (dc_pv <= 1).all(), "P-values must be in range [0, 1]"
    
    # Test that results are finite
    assert np.isfinite(dc_es).all(), "ES values must be finite"
    assert np.isfinite(dc_pv).all(), "P-values must be finite"
    
    # Compare results with reasonable tolerance
    max_diff = (gp_es - dc_es).abs().values.max()
    assert max_diff < 0.10, f"Maximum difference {max_diff} exceeds tolerance of 0.10"
    

