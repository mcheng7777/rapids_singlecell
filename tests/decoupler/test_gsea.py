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
    times = 10  # Reduced for faster testing
    seed = 42
    X, obs, var = mat
    gene_sets = net.groupby("source")["target"].apply(lambda x: list(x)).to_dict()
    cnct, starts, offsets = idxmat
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



