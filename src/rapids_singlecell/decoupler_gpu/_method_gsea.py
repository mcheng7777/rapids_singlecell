"""
Optimized GSEA implementation using CuPy RawKernels for maximum performance.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
from tqdm import tqdm

from ._helper._data import _log
from ._helper._net import _getset
from ._helper._Method import Method, MethodMeta
from ._method_aucell import rank_rows_desc



# =============================================================================
# CUDA KERNELS FOR GSEA
# =============================================================================

# GSEA kernel should be ES and NES for one gene set and cell
# NES executes ES function across permutations
# stsgsea executes the GSEA kernel for each cell x gene set
# Grid: (nsrc, grid_y) 
# - nsrc: number of gene sets
# - grid_y = (nobs + threads_per_block - 1)//threads_per_block
# Block: (threads_per_block,)
# threads_per_block = 32


# # ES computation kernel - processes multiple sources in parallel
# _es_kernel = cp.RawKernel(
#     r"""
# extern "C" __global__
# void es_kernel(
#     const float* __restrict__ row,
#     const int* __restrict__ rnks,
#     const bool* __restrict__ set_msk_batch,
#     const float* __restrict__ dec_batch,
#     float* __restrict__ es_results,
#     int* __restrict__ es_indices,
#     const int n_sources,
#     const int n_features
# ) {
#     const int set_idx = blockIdx.x
#     const int idx  = blockIDx.y * blockDim.x + threadIdx.x
#     if (source_idx >= n_sources) return;
    
#     // Get source-specific data
#     const bool* set_msk = set_msk_batch + source_idx * n_features;
#     const float dec = dec_batch[source_idx];
    
#     // Compute sum_set (sum of absolute values in set)
#     float sum_set = 0.0f;
#     for (int k = 0; k < n_features; k++) {
#         if (set_msk[k]) {
#             float val = row[k];
#             sum_set += (val >= 0.0f) ? val : -val;
#         }
#     }
    
#     if (sum_set == 0.0f) {
#         es_results[source_idx] = 0.0f;
#         es_indices[source_idx] = 0;
#         return;
#     }
    
#     // Compute ES
#     float cum_sum = 0.0f;
#     float mx_pos = 0.0f;
#     float mx_neg = 0.0f;
#     int j_pos = 0;
#     int j_neg = 0;
    
#     for (int i = 0; i < n_features; i++) {
#         int rank_idx = rnks[i];
#         if (set_msk[rank_idx]) {
#             float val = row[rank_idx];
#             cum_sum += ((val >= 0.0f) ? val : -val) / sum_set;
#         } else {
#             cum_sum -= dec;
#         }
        
#         if (cum_sum > mx_pos) {
#             mx_pos = cum_sum;
#             j_pos = rank_idx;
#         }
#         if (cum_sum < mx_neg) {
#             mx_neg = cum_sum;
#             j_neg = rank_idx;
#         }
#     }
    
#     if (mx_pos > -mx_neg) {
#         es_results[source_idx] = mx_pos;
#         es_indices[source_idx] = j_pos;
#     } else {
#         es_results[source_idx] = mx_neg;
#         es_indices[source_idx] = j_neg;
#     }
# }
# """,
#     "es_kernel",
# )


# # NES computation kernel - processes multiple sources with permutations
# _nes_kernel = cp.RawKernel(
#     r"""
# extern "C" __global__
# void nes_kernel(
#     const int* __restrict__ ridx,
#     const float* __restrict__ row,
#     const int* __restrict__ rnks,
#     const bool* __restrict__ set_msk_batch,
#     const float* __restrict__ dec_batch,
#     const float* __restrict__ es_batch,
#     float* __restrict__ nes_results,
#     float* __restrict__ pv_results,
#     const int n_sources,
#     const int n_features,
#     const int n_permutations
# ) {
#     const int source_idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if (source_idx >= n_sources) return;
    
#     // Get source-specific data
#     const bool* set_msk = set_msk_batch + source_idx * n_features;
#     const float dec = dec_batch[source_idx];
#     const float es = es_batch[source_idx];
    
#     // Compute null distribution
#     float null_sum_pos = 0.0f;
#     float null_sum_neg = 0.0f;
#     int null_count_pos = 0;
#     int null_count_neg = 0;
#     int pos_ge_es_count = 0;
#     int neg_le_es_count = 0;
    
#     for (int perm = 0; perm < n_permutations; perm++) {
#         // Get permuted mask
#         bool permuted_msk[1024];  // Max features - adjust if needed
#         for (int k = 0; k < n_features; k++) {
#             permuted_msk[k] = set_msk[ridx[perm * n_features + k]];
#         }
        
#         // Compute ES for this permutation
#         float sum_set = 0.0f;
#         for (int k = 0; k < n_features; k++) {
#             if (permuted_msk[k]) {
#                 float val = row[k];
#                 sum_set += (val >= 0.0f) ? val : -val;
#             }
#         }
        
#         if (sum_set == 0.0f) continue;
        
#         float cum_sum = 0.0f;
#         float mx_pos = 0.0f;
#         float mx_neg = 0.0f;
        
#         for (int i = 0; i < n_features; i++) {
#             int rank_idx = rnks[i];
#             if (permuted_msk[rank_idx]) {
#                 float val = row[rank_idx];
#                 cum_sum += ((val >= 0.0f) ? val : -val) / sum_set;
#             } else {
#                 cum_sum -= dec;
#             }
            
#             if (cum_sum > mx_pos) mx_pos = cum_sum;
#             if (cum_sum < mx_neg) mx_neg = cum_sum;
#         }
        
#         float null_es = (mx_pos > -mx_neg) ? mx_pos : mx_neg;
        
#         if (null_es >= 0.0f) {
#             null_sum_pos += null_es;
#             null_count_pos++;
#             if (null_es >= es) pos_ge_es_count++;
#         } else {
#             null_sum_neg += null_es;
#             null_count_neg++;
#             if (null_es <= es) neg_le_es_count++;
#         }
#     }
    
#     // Compute NES and p-value
#     if (es >= 0.0f && null_count_pos > 0) {
#         float pos_null_mean = null_sum_pos / null_count_pos;
#         nes_results[source_idx] = es / pos_null_mean;
#         pv_results[source_idx] = (float)pos_ge_es_count / null_count_pos;
#     } else if (es < 0.0f && null_count_neg > 0) {
#         float neg_null_mean = null_sum_neg / null_count_neg;
#         nes_results[source_idx] = -es / neg_null_mean;
#         pv_results[source_idx] = (float)neg_le_es_count / null_count_neg;
#     } else {
#         nes_results[source_idx] = 0.0f;
#         pv_results[source_idx] = 1.0f;
#     }
# }
# """,
#     "nes_kernel",
# )

_gsea_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void gsea_kernel(
    const float* __restrict__ mat,
    const bool* __restrict__ set_msks,
    const float* __restrict__ decs,
    const int* __restrict__ ridx,
    const int* __restrict__ rnks,
    float* __restrict__ es,
    float* __restrict__ nes,
    float* __restrict__ pv,
    const int n_obs,
    const int n_sets,
    const int n_features,
    const int n_permutations
) {
    const int set_idx = blockIdx.x;
    const int obs_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (obs_idx >= n_obs || set_idx >= n_sets) return;
    
    const bool* set_msk = set_msks + set_idx * n_features;
    
    const float dec = decs[set_idx];

    // ES section
    // compute norm
    float sum_set = 0.0f;
    for (int k = 0; k < n_features; k++) {
        if (set_msk[k]) {
            float val = mat[obs_idx * n_features + k];
            sum_set += (val >= 0.0f) ? val : -val;
        }
    }

    
    // compute ES
    float true_es = 0.0f;
    float cum_sum = 0.0f;
    float mx_pos = 0.0f;
    float mx_neg = 0.0f;
    int j_pos = 0;
    int j_neg = 0;

    if (sum_set == 0.0f) {
        true_es = 0.0f;
    } else {
        for (int i = 0; i < n_features; i++) {
            int rank_idx = rnks[obs_idx * n_features + i];
            if (set_msk[rank_idx]) {
                float val = mat[obs_idx * n_features + rank_idx];
                cum_sum += ((val >= 0.0f) ? val : -val) / sum_set;   
            } else {
                cum_sum -= dec;
            }
            // update max positive and negative deviations
            if (cum_sum > mx_pos) {
                mx_pos = cum_sum;
                j_pos = rank_idx;
            }
            if (cum_sum < mx_neg) {
                mx_neg = cum_sum;
                j_neg = rank_idx;
            }
        }
        // determine if pos or neg are more enriched
        if (mx_pos > -mx_neg) {
            true_es = mx_pos;
        } else {
            true_es = mx_neg;
        }
    }


    // NES section
    if (n_permutations == 0){
        es[obs_idx * n_sets + set_idx] = true_es;
        nes[obs_idx * n_sets + set_idx] = 0.0f;
        pv[obs_idx * n_sets + set_idx] = 1.0f;
    } else {
        // compute null distribution
        float null_sum_pos = 0.0f;
        float null_sum_neg = 0.0f;
        int null_count_pos = 0;
        int null_count_neg = 0;
        int pos_ge_es_count = 0;
        int neg_le_es_count = 0;
        for (int perm = 0; perm < n_permutations; perm++) {
            // get permuted mask
            //bool permuted_msk[n_features];
            //for (int k = 0; k < n_features; k++) {
            //    permuted_msk[k] = set_msk[ridx[perm * n_features + k]];
            //}
            
            // compute null sum
            float perm_sum_set = 0.0f;
            float perm_cum_sum = 0.0f;
            float perm_mx_pos = 0.0f;
            float perm_mx_neg = 0.0f;
            float perm_mx_value = 0.0f;
            for (int k = 0; k < n_features; k++) {
                if (set_msk[ridx[perm * n_features + k]]) {
                    float val = mat[obs_idx * n_features + k];
                    perm_sum_set += (val >= 0.0f) ? val : -val;
                }
            }
            if (perm_sum_set == 0.0f){
                // treat as positive null
                null_count_pos++;
            } else {
                // compute null ES
                for (int i = 0; i < n_features; i++) {
                    int rank_idx = rnks[obs_idx * n_features + i];
                    if (set_msk[ridx[perm * n_features + rank_idx]]) {
                        float val = mat[obs_idx * n_features + rank_idx];
                        perm_cum_sum += ((val >= 0.0f) ? val : -val) / perm_sum_set;
                    } else {
                        perm_cum_sum -= dec;
                    }
                    if (perm_cum_sum > perm_mx_pos) {
                        perm_mx_pos = perm_cum_sum;
                    }
                    if (perm_cum_sum < perm_mx_neg) {
                        perm_mx_neg = perm_cum_sum;
                        perm_mx_value = perm_cum_sum;
                    }
                }
                // update null sum and counts
                perm_mx_value = (perm_mx_pos > -perm_mx_neg) ? perm_mx_pos : perm_mx_neg;
                if (perm_mx_value >= 0.0f){
                    null_sum_pos += perm_mx_value;
                    null_count_pos++;
                    if (perm_mx_value >= true_es) pos_ge_es_count++;
                } else {
                    null_sum_neg += perm_mx_value;
                    null_count_neg++;
                    if (perm_mx_value <= true_es) neg_le_es_count++;
                }
            }
        }
        // compute NES and p-value
        float pval = 1.0f;
        float nes_val = 0.0f;

        if (true_es >= 0.0f && null_count_pos > 0){
            pval = (float)pos_ge_es_count / null_count_pos;
            float pos_null_mean = null_sum_pos / null_count_pos;
            nes_val = true_es / pos_null_mean;
        } else if (true_es < 0.0f && null_count_neg > 0){
            pval = (float)neg_le_es_count / null_count_neg;
            float neg_null_mean = null_sum_neg / null_count_neg;
            nes_val = -true_es / neg_null_mean;
        } else {
            nes_val = 0.0f;
            pval = 1.0f;
        }

        // write to output
        es[obs_idx * n_sets + set_idx] = true_es;
        nes[obs_idx * n_sets + set_idx] = nes_val;
        pv[obs_idx * n_sets + set_idx] = pval;
    }
}
""",
"gsea_kernel"
)


# # Batch processing kernel - processes multiple observations and sources
# _batch_gsea_kernel = cp.RawKernel(
#     r"""
# extern "C" __global__
# void batch_gsea_kernel(
#     const float* __restrict__ mat,
#     const bool* __restrict__ set_msk_batch,
#     const float* __restrict__ dec_batch,
#     const int* __restrict__ ridx,
#     const int* __restrict__ rnks_2d,
#     float* __restrict__ es_results,
#     float* __restrict__ nes_results,
#     float* __restrict__ pv_results,
#     const int n_obs,
#     const int n_sources,
#     const int n_features,
#     const int n_permutations
# ) {
#     const int obs_idx = blockIdx.x;
#     const int source_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
#     if (obs_idx >= n_obs || source_idx >= n_sources) return;
    
#     // Get observation data
#     const float* row = mat + obs_idx * n_features;
    
#     // Get source data
#     const bool* set_msk = set_msk_batch + source_idx * n_features;
#     const float dec = dec_batch[source_idx];
    
#     // Compute ES (inline)
#     float sum_set = 0.0f;
#     for (int k = 0; k < n_features; k++) {
#         if (set_msk[k]) {
#             float val = row[k];
#             sum_set += (val >= 0.0f) ? val : -val;
#         }
#     }
    
#     if (sum_set == 0.0f) {
#         es_results[obs_idx * n_sources + source_idx] = 0.0f;
#         nes_results[obs_idx * n_sources + source_idx] = 0.0f;
#         pv_results[obs_idx * n_sources + source_idx] = 1.0f;
#         return;
#     }
    
#     float cum_sum = 0.0f;
#     float mx_pos = 0.0f;
#     float mx_neg = 0.0f;
#     int j_pos = 0;
#     int j_neg = 0;
    
#     for (int i = 0; i < n_features; i++) {
#         int rank_idx = rnks_2d[obs_idx * n_features + i];
#         if (set_msk[rank_idx]) {
#             float val = row[rank_idx];
#             cum_sum += ((val >= 0.0f) ? val : -val) / sum_set;
#         } else {
#             cum_sum -= dec;
#         }
        
#         // Store cumulative sum at this position (like decoupler es[i] = cum_sum)
#         es_results[rank_idx] = cum_sum;
        
#         // Update maximum positive and negative deviations
#         if (cum_sum > mx_pos) {
#             mx_pos = cum_sum;
#             j_pos = rank_idx;  // Use rank_idx (feature index) not i (loop index)
#         }
#         if (cum_sum < mx_neg) {
#             mx_neg = cum_sum;
#             j_neg = rank_idx;  // Use rank_idx (feature index) not i (loop index)
#         }
#     }
    
#     float es = (mx_pos > -mx_neg) ? mx_pos : mx_neg;
#     es_results[obs_idx * n_sources + source_idx] = es;
    
#     // Compute NES with permutations
#     if (n_permutations > 0) {
#         // Compute null distribution
#         float null_sum_pos = 0.0f;
#         float null_sum_neg = 0.0f;
#         int null_count_pos = 0;
#         int null_count_neg = 0;
#         int pos_ge_es_count = 0;
#         int neg_le_es_count = 0;
        
#         for (int perm = 0; perm < n_permutations; perm++) {
#             // Get permuted mask
#             bool permuted_msk[1024];  // Max features - adjust if needed
#             for (int k = 0; k < n_features; k++) {
#                 permuted_msk[k] = set_msk[ridx[perm * n_features + k]];
#             }
            
#             // Compute ES for this permutation
#             float perm_sum_set = 0.0f;
#             for (int k = 0; k < n_features; k++) {
#                 if (permuted_msk[k]) {
#                     float val = row[k];
#                     perm_sum_set += (val >= 0.0f) ? val : -val;
#                 }
#             }
            
#             if (perm_sum_set == 0.0f) continue;
            
#             float perm_cum_sum = 0.0f;
#             float perm_mx_pos = 0.0f;
#             float perm_mx_neg = 0.0f;
            
#             for (int i = 0; i < n_features; i++) {
#                 int rank_idx = rnks_2d[obs_idx * n_features + i];
#                 if (permuted_msk[rank_idx]) {
#                     float val = row[rank_idx];
#                     perm_cum_sum += ((val >= 0.0f) ? val : -val) / perm_sum_set;
#                 } else {
#                     perm_cum_sum -= dec;
#                 }
                
#                 // Update maximum positive and negative deviations
#                 if (perm_cum_sum > perm_mx_pos) perm_mx_pos = perm_cum_sum;
#                 if (perm_cum_sum < perm_mx_neg) perm_mx_neg = perm_cum_sum;
#             }
            
#             float null_es = (perm_mx_pos > -perm_mx_neg) ? perm_mx_pos : perm_mx_neg;
            
#             if (null_es >= 0.0f) {
#                 null_sum_pos += null_es;
#                 null_count_pos++;
#                 if (null_es >= es) pos_ge_es_count++;
#             } else {
#                 null_sum_neg += null_es;
#                 null_count_neg++;
#                 if (null_es <= es) neg_le_es_count++;
#             }
#         }
        
#         // Compute NES and p-value
#         if (es >= 0.0f && null_count_pos > 0) {
#             float pos_null_mean = null_sum_pos / null_count_pos;
#             nes_results[obs_idx * n_sources + source_idx] = es / pos_null_mean;
#             pv_results[obs_idx * n_sources + source_idx] = (float)pos_ge_es_count / null_count_pos;
#         } else if (es < 0.0f && null_count_neg > 0) {
#             float neg_null_mean = null_sum_neg / null_count_neg;
#             nes_results[obs_idx * n_sources + source_idx] = -es / neg_null_mean;
#             pv_results[obs_idx * n_sources + source_idx] = (float)neg_le_es_count / null_count_neg;
#         } else {
#             nes_results[obs_idx * n_sources + source_idx] = 0.0f;
#             pv_results[obs_idx * n_sources + source_idx] = 1.0f;
#         }
#     } else {
#         nes_results[obs_idx * n_sources + source_idx] = es;
#         pv_results[obs_idx * n_sources + source_idx] = 1.0f;
#     }
# }
# """,
#     "batch_gsea_kernel",
# )


# =============================================================================
# KERNEL LAUNCH FUNCTIONS
# =============================================================================

def _stgsea(mat, cnct, starts, offsets, times, seed, verbose) -> tuple[cp.ndarray, cp.ndarray]:
    """Use the full batch kernel for maximum parallelism."""
    nobs, nvar = mat.shape
    nsrc = starts.size
    
    # Pre-compute all feature sets and masks
    set_msk_batch = cp.zeros((nsrc, nvar), dtype=cp.bool_)
    dec_batch = cp.zeros(nsrc, dtype=cp.float32)
    
    for j in range(nsrc):
        fset = _getset(cnct, starts, offsets, j)
        set_msk_batch[j, fset] = True
        dec_batch[j] = 1.0 / (nvar - fset.size)
    
    # Generate permutation indices
    if times > 1:
        ridx = _ridx(times=times, nvar=nvar, seed=seed)
    else:
        ridx = cp.zeros((1, nvar), dtype=cp.int32)
        ridx[0] = cp.arange(nvar)
    
    # Compute per-observation ranks (descending by value)
    # rnks[i, :] gives feature indices sorted for observation i
    rnks = rank_rows_desc(mat)

    # Launch batch kernel with per-observation ranks
    es, nes, pv = launch_batch_gsea_kernel(
        mat, set_msk_batch, dec_batch, ridx, rnks,
        nobs, nsrc, nvar, times
    )
    
    return es, nes, pv
# def _stsgsea(mat, cnct, starts, offsets, times, seed, verbose) -> tuple[cp.ndarray, cp.ndarray]:
#     nobs, nvar = mat.shape
#     nsrc = starts.size
#     # Sort features
#     ranks = rank_rows_desc(mat)
#     R, C = ranks.shape
#     es = cp.zeros((R, nsrc), dtype=cp.float32)
#     nes = cp.zeros((R, nsrc), dtype=cp.float32)
#     pv = cp.zeros((R, nsrc), dtype=cp.float32)
    
#     threads_per_block = 32
#     grid_y = (R + threads_per_block - 1) // threads_per_block
#     blocks_y = (nsrc + threads_per_block - 1) // threads_per_block
    
#     _batch_gsea_kernel(
#         (nsrc, grid_y), (threads_per_block,),
#         (mat, set_msk_batch, dec_batch, ridx, rnks_2d,
#          es, nes, pv,
#          n_obs, n_sources, n_features, n_permutations)
#     )
#     return es, pv

# def launch_es_kernel(row, rnks, set_msk_batch, dec_batch, n_sources, n_features):
#     """Launch ES computation kernel for multiple sources."""
#     es_results = cp.zeros(n_sources, dtype=cp.float32)
#     es_indices = cp.zeros(n_sources, dtype=cp.int32)
    
#     threads_per_block = 256
#     blocks = (n_sources + threads_per_block - 1) // threads_per_block
    
#     _es_kernel(
#         (blocks,), (threads_per_block,),
#         (row, rnks, set_msk_batch, dec_batch, es_results, es_indices,
#          n_sources, n_features)
#     )
    
#     return es_results, es_indices


# def launch_nes_kernel(ridx, row, rnks, set_msk_batch, dec_batch, es_batch, 
#                      n_sources, n_features, n_permutations):
#     """Launch NES computation kernel for multiple sources."""
#     nes_results = cp.zeros(n_sources, dtype=cp.float32)
#     pv_results = cp.zeros(n_sources, dtype=cp.float32)
    
#     threads_per_block = 256
#     blocks = (n_sources + threads_per_block - 1) // threads_per_block
    
#     _nes_kernel(
#         (blocks,), (threads_per_block,),
#         (ridx, row, rnks, set_msk_batch, dec_batch, es_batch,
#          nes_results, pv_results, n_sources, n_features, n_permutations)
#     )
    
#     return nes_results, pv_results


def launch_batch_gsea_kernel(mat, set_msk_batch, dec_batch, ridx, rnks_2d,
                            n_obs, n_sources, n_features, n_permutations):
    """Launch batch GSEA kernel for multiple observations and sources."""
    es_results = cp.zeros((n_obs, n_sources), dtype=cp.float32)
    nes_results = cp.zeros((n_obs, n_sources), dtype=cp.float32)
    pv_results = cp.zeros((n_obs, n_sources), dtype=cp.float32)
    
    threads_per_block = 32
    blocks_x = n_sources
    blocks_y = (n_obs + threads_per_block - 1) // threads_per_block
    
    _gsea_kernel(
        (blocks_x, blocks_y), (threads_per_block, 1),
        (mat, set_msk_batch, dec_batch, ridx, rnks_2d,
         es_results, nes_results, pv_results,
         n_obs, n_sources, n_features, n_permutations)
    )
    
    return es_results, nes_results, pv_results



def _ridx(
    times: int,
    nvar: int,
    seed: int | None,
):
    idx = cp.tile(cp.arange(nvar), (times, 1))
    if seed:
        rng = cp.random.RandomState(seed=seed)
        for i in idx:
            rng.shuffle(i)
    return idx


def _func_gsea(
    mat: cp.ndarray,
    cnct: cp.ndarray,
    starts: cp.ndarray,
    offsets: cp.ndarray,
    *,
    times: int | float = 1000,
    seed: int | float = 42,
    verbose: bool = False,
    use_batch_kernel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimized Gene Set Enrichment Analysis using CuPy RawKernels.
    
    This implementation provides two optimization strategies:
    1. Batch kernel processing (use_batch_kernel=True) - processes all observations and sources in parallel
    2. Source batching (use_batch_kernel=False) - processes sources in parallel per observation
    
    Parameters
    ----------
    mat : cp.ndarray
        Expression matrix (n_obs x n_features)
    cnct : cp.ndarray
        Connection indices for feature sets
    starts : cp.ndarray
        Start indices for each feature set
    offsets : cp.ndarray
        Offset indices for each feature set
    times : int, default=1000
        Number of permutations for p-value calculation
    seed : int, default=42
        Random seed for permutations
    verbose : bool, default=False
        Whether to show progress
    use_batch_kernel : bool, default=True
        Whether to use the full batch kernel (faster but more memory intensive)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (es, pv) matrices of shape (n_obs, n_sources)
    """
    nobs, nvar = mat.shape
    assert isinstance(times, int | float) and times >= 0, "times must be numeric and >= 0"
    assert isinstance(seed, int | float) and seed >= 0, "seed must be numeric and >= 0"
    times, seed = int(times), int(seed)

    # Compute
    nsrc = starts.size    
    m = f"gsea_optimized - calculating {nsrc} scores across {nobs} observations"
    _log(m, level="info", verbose=verbose)
    if times > 1:
        m = f"gsea_optimized - comparing estimates against {times} random permutations"
        _log(m, level="info", verbose=verbose)

    es, nes, pv = _batch_kernel_approach(mat, cnct, starts, offsets, times, seed, verbose)
    if times > 1:
        es = nes
    
    # if use_batch_kernel and nobs * nsrc < 1_000_000:  # Memory threshold
    #     es, pv = _batch_kernel_approach(mat, cnct, starts, offsets, times, seed, verbose)
    # else:
    #     es, pv = _source_batching_approach(mat, cnct, starts, offsets, times, seed, verbose)
    
    return es.get(), pv.get()


def _batch_kernel_approach(mat, cnct, starts, offsets, times, seed, verbose) -> tuple[cp.ndarray, cp.ndarray]:
    """Use the full batch kernel for maximum parallelism."""
    nobs, nvar = mat.shape
    nsrc = starts.size
    
    # Pre-compute all feature sets and masks
    set_msk_batch = cp.zeros((nsrc, nvar), dtype=cp.bool_)
    dec_batch = cp.zeros(nsrc, dtype=cp.float32)
    
    for j in range(nsrc):
        fset = _getset(cnct, starts, offsets, j)
        set_msk_batch[j, fset] = True
        dec_batch[j] = 1.0 / (nvar - fset.size)
    
    # Generate permutation indices
    if times > 1:
        ridx = _ridx(times=times, nvar=nvar, seed=seed)
    else:
        ridx = cp.zeros((1, nvar), dtype=cp.int32)
        ridx[0] = cp.arange(nvar)
    
    # Compute per-observation ranks (descending by value)
    # rnks_2d[i, :] gives feature indices sorted for observation i
    rnks_2d = cp.argsort(-mat, axis=1).astype(cp.int32)

    # Launch batch kernel with per-observation ranks
    es, nes, pv = launch_batch_gsea_kernel(
        mat, set_msk_batch, dec_batch, ridx, rnks_2d,
        nobs, nsrc, nvar, times
    )
    
    
    # Return all three arrays (ES, NES, PV)
    return es, nes, pv


# def _source_batching_approach(mat, cnct, starts, offsets, times, seed, verbose) -> tuple[cp.ndarray, cp.ndarray]:
#     """Use source batching for memory-efficient processing."""
#     nobs, nvar = mat.shape
#     nsrc = starts.size
    
#     # Pre-compute all feature sets and masks
#     set_msk_batch = cp.zeros((nsrc, nvar), dtype=cp.bool_)
#     dec_batch = cp.zeros(nsrc, dtype=cp.float32)
    
#     for j in range(nsrc):
#         fset = _getset(cnct, starts, offsets, j)
#         set_msk_batch[j, fset] = True
#         dec_batch[j] = 1.0 / (nvar - fset.size)
    
#     # Generate permutation indices
#     if times > 1:
#         ridx = _ridx(times=times, nvar=nvar, seed=seed)
#     else:
#         ridx = cp.zeros((1, nvar), dtype=cp.int32)
#         ridx[0] = cp.arange(nvar)
    
#     # Initialize results
#     es = cp.zeros((nobs, nsrc), dtype=cp.float32)
#     nes = cp.zeros((nobs, nsrc), dtype=cp.float32)
#     pv = cp.zeros((nobs, nsrc), dtype=cp.float32)
    
#     # Process each observation
#     for i in tqdm(range(nobs), disable=not verbose):
#         row = mat[i]
        
#         # Sort features by expression (descending)
#         idx = cp.argsort(-row)
#         row_sorted = row[idx]
#         set_msk_sorted = set_msk_batch[:, idx]
#         rnks = cp.arange(nvar, dtype=cp.int32)
        
#         # Compute ES for all sources in parallel
#         es_batch, _ = launch_es_kernel(
#             row_sorted, rnks, set_msk_sorted, dec_batch, nsrc, nvar
#         )
        
#         # Compute NES for all sources in parallel
#         if times > 1:
#             nes_batch, pv_batch = launch_nes_kernel(
#                 ridx, row_sorted, rnks, set_msk_sorted, dec_batch, es_batch,
#                 nsrc, nvar, times
#             )
#         else:
#             nes_batch = es_batch
#             pv_batch = cp.ones(nsrc, dtype=cp.float32)
        
#         es[i] = es_batch
#         nes[i] = nes_batch
#         pv[i] = pv_batch
    
#     return nes, pv


# def _func_gsea_memory_optimized(
#     mat: cp.ndarray,
#     cnct: cp.ndarray,
#     starts: cp.ndarray,
#     offsets: cp.ndarray,
#     *,
#     times: int | float = 1000,
#     seed: int | float = 42,
#     verbose: bool = False,
#     batch_size: int = 100,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Memory-optimized GSEA using CuPy kernels with batching.
    
#     This version processes observations in batches to manage GPU memory usage.
#     """
#     nobs, nvar = mat.shape
#     nsrc = starts.size
    
#     # Validate inputs
#     assert isinstance(times, int | float) and times >= 0, "times must be numeric and >= 0"
#     assert isinstance(seed, int | float) and seed >= 0, "seed must be numeric and >= 0"
#     times, seed = int(times), int(seed)
    
#     # Pre-compute feature sets
#     set_msk_batch = cp.zeros((nsrc, nvar), dtype=cp.bool_)
#     dec_batch = cp.zeros(nsrc, dtype=cp.float32)
    
#     for j in range(nsrc):
#         fset = _getset(cnct, starts, offsets, j)
#         set_msk_batch[j, fset] = True
#         dec_batch[j] = 1.0 / (nvar - fset.size)
    
#     # Generate permutation indices
#     if times > 1:
#         ridx = _ridx(times=times, nvar=nvar, seed=seed)
#     else:
#         ridx = cp.zeros((1, nvar), dtype=cp.int32)
#         ridx[0] = cp.arange(nvar)
    
#     # Initialize results
#     es = cp.zeros((nobs, nsrc), dtype=cp.float32)
#     nes = cp.zeros((nobs, nsrc), dtype=cp.float32)
#     pv = cp.zeros((nobs, nsrc), dtype=cp.float32)
    
#     # Process in batches
#     for start_idx in tqdm(range(0, nobs, batch_size), disable=not verbose):
#         end_idx = min(start_idx + batch_size, nobs)
#         batch_mat = mat[start_idx:end_idx]
#         batch_nobs = batch_mat.shape[0]
        
#         # Process this batch
#         for i in range(batch_nobs):
#             row = batch_mat[i]
            
#             # Sort features
#             idx = cp.argsort(-row)
#             row_sorted = row[idx]
#             set_msk_sorted = set_msk_batch[:, idx]
#             rnks = cp.arange(nvar, dtype=cp.int32)
            
#             # Compute ES and NES
#             es_batch, _ = launch_es_kernel(
#                 row_sorted, rnks, set_msk_sorted, dec_batch, nsrc, nvar
#             )
            
#             if times > 1:
#                 nes_batch, pv_batch = launch_nes_kernel(
#                     ridx, row_sorted, rnks, set_msk_sorted, dec_batch, es_batch,
#                     nsrc, nvar, times
#                 )
#             else:
#                 nes_batch = es_batch
#                 pv_batch = cp.ones(nsrc, dtype=cp.float32)
            
#             es[start_idx + i] = es_batch
#             nes[start_idx + i] = nes_batch
#             pv[start_idx + i] = pv_batch
    
#     return nes.get(), pv.get()


# =============================================================================
# PERFORMANCE COMPARISON UTILITIES
# =============================================================================

def benchmark_gsea_implementations(
    mat: cp.ndarray,
    cnct: cp.ndarray,
    starts: cp.ndarray,
    offsets: cp.ndarray,
    times: int = 100,
    seed: int = 42,
) -> dict:
    """
    Benchmark different GSEA implementations.
    
    Returns timing results for comparison.
    """
    import time
    
    results = {}
    
    # Test original implementation
    from ._method_gsea import _func_gsea as _func_gsea_original
    
    start_time = time.time()
    es_orig, pv_orig = _func_gsea_original(
        mat, cnct, starts, offsets, times=times, seed=seed, verbose=False
    )
    results['original'] = time.time() - start_time
    
    # Test optimized implementation
    start_time = time.time()
    es_opt, pv_opt = _func_gsea(
        mat, cnct, starts, offsets, times=times, seed=seed, verbose=False
    )
    results['optimized'] = time.time() - start_time
    
    # Test memory-optimized implementation
    start_time = time.time()
    es_mem, pv_mem = _func_gsea_memory_optimized(
        mat, cnct, starts, offsets, times=times, seed=seed, verbose=False
    )
    results['memory_optimized'] = time.time() - start_time
    
    # Verify results are similar
    results['es_diff_orig_opt'] = cp.max(cp.abs(es_orig - es_opt))
    results['pv_diff_orig_opt'] = cp.max(cp.abs(pv_orig - pv_opt))
    results['es_diff_orig_mem'] = cp.max(cp.abs(es_orig - es_mem))
    results['pv_diff_orig_mem'] = cp.max(cp.abs(pv_orig - pv_mem))
    
    return results


# =============================================================================
# METHOD CLASS AND METADATA
# =============================================================================

class GseaMethod(Method):
    """Custom Method class for optimized GSEA with enhanced performance options."""

    def __call__(
        self,
        data,
        net,
        *,
        tmin: int | float = 5,
        raw: bool = False,
        empty: bool = True,
        bsize: int | float = 5000,
        verbose: bool = False,
        pre_load: bool = False,
        adj_pv_gpu: bool = False,
        use_batch_kernel: bool = True,
        **kwargs,
    ):
        from ._helper._run import _run
        
        return _run(
            name=self.name,
            func=self.func,
            adj=self.adj,
            test=self.test,
            data=data,
            net=net,
            tmin=tmin,
            raw=raw,
            empty=empty,
            bsize=bsize,
            verbose=verbose,
            pre_load=pre_load,
            adj_pv_gpu=adj_pv_gpu,
            use_batch_kernel=use_batch_kernel,
            **kwargs,
        )


_gsea = MethodMeta(
    name="gsea",
    desc="Gene Set Enrichment Analysis (GSEA)",
    func=_func_gsea,
    stype="numerical",
    adj=False,
    weight=False,
    test=True,
    limits=(-cp.inf, +cp.inf),
    reference="https://doi.org/10.1073/pnas.0506580102",
)

gsea = GseaMethod(_method=_gsea)
