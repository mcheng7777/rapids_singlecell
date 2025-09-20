from __future__ import annotations

import cupy as cp
import numpy as np
from tqdm import tqdm

from ._helper._data import _log
from ._helper._net import _getset
from ._helper._Method import Method, MethodMeta
# from ._method_aucell import rank_rows_desc



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
    if (n_permutations <= 1){
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
                perm_mx_value = 0.0f;
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
                    }
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


def rank_rows_desc(x: cp.ndarray) -> cp.ndarray:
    return cp.argsort(-x, axis=1)

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

def _stgsea(mat, cnct, starts, offsets, ridx, times, seed, verbose) -> tuple[cp.ndarray, cp.ndarray]:
    """Use the full batch kernel for maximum parallelism."""
    # Compute per-observation ranks (descending by value)
    # rnks[i, :] gives feature indices sorted for observation i
    rnks = rank_rows_desc(mat)
    
    nobs, nvar = mat.shape
    nsrc = starts.size
    
    # Pre-compute all feature sets and masks
    set_msk_batch = cp.zeros((nsrc, nvar), dtype=cp.bool_)
    dec_batch = cp.zeros(nsrc, dtype=cp.float32)
    
    for j in range(nsrc):
        fset = _getset(cnct, starts, offsets, j)
        set_msk_batch[j, fset] = True
        dec_batch[j] = 1.0 / (nvar - fset.size)
    
    es = cp.zeros((n_obs, n_sources), dtype=cp.float32)
    nes = cp.zeros((n_obs, n_sources), dtype=cp.float32)
    pv = cp.zeros((n_obs, n_sources), dtype=cp.float32)

    threads_per_block = 32
    blocks_x = n_sources
    blocks_y = (n_obs + threads_per_block - 1) // threads_per_block


    _gsea_kernel(
        (blocks_x, blocks_y), 
        (threads_per_block, 1),
        (mat, set_msk_batch, dec_batch, ridx, rnks, es, nes, pv, n_obs, n_sources, n_features, times)
    )
    
    return es, nes, pv

# def launch_batch_gsea_kernel(mat, set_msk_batch, dec_batch, ridx, rnks_2d,
#                             n_obs, n_sources, n_features, n_permutations):
#     """Launch batch GSEA kernel for multiple observations and sources."""
#     es_results = cp.zeros((n_obs, n_sources), dtype=cp.float32)
#     nes_results = cp.zeros((n_obs, n_sources), dtype=cp.float32)
#     pv_results = cp.zeros((n_obs, n_sources), dtype=cp.float32)
    
#     threads_per_block = 32
#     blocks_x = n_sources
#     blocks_y = (n_obs + threads_per_block - 1) // threads_per_block
    
#     _gsea_kernel(
#         (blocks_x, blocks_y), (threads_per_block, 1),
#         (mat, set_msk_batch, dec_batch, ridx, rnks_2d,
#          es_results, nes_results, pv_results,
#          n_obs, n_sources, n_features, n_permutations)
#     )
    
#     return es_results, nes_results, pv_results


def _func_gsea(
    mat: cp.ndarray,
    cnct: cp.ndarray,
    starts: cp.ndarray,
    offsets: cp.ndarray,
    *,
    times: int | float = 1000,
    seed: int | float = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Gene Set Enrichment Analysis (GSEA) :cite:`gsea`.

    Features are ranked based on a continuous statistic (e.g., expression, score, or correlation).
    The enrichment score (ES) for a feature set is computed by walking down the ranked list and increasing a running-sum
    statistic when a feature is in the set, and decreasing it when it is not.

    .. math::

       \delta(F, i) =
       \begin{cases}
       \frac{|r_i|}{\sum\limits_{j \in F} |r_j|} & \text{if feature } i \in F \\
       -\frac{1}{l} & \text{if feature } i \notin F
       \end{cases}

    Where:

    - :math:`F` is a feature set
    - :math:`r` is the ranking of the feature statistics in descending order
    - :math:`r_i` is the value for feature :math:`i`
    - :math:`r_j` is the value for feature :math:`j` in :math:`F`
    - :math:`k` is the number of features in :math:`F`
    - :math:`N` is the total number of features in :math:`r`
    - :math:`l=N-k` is the number of features not in :math:`F` but present in :math:`r`

    For each feature, the function :math:`\delta(F,i)` is applied and stored as a sequence :math:`L`.

    .. math::

        L = \delta(F, i)\text{ for i} = \text{1, 2, ... , N}

    The enrichment score :math:`ES` corresponds to the maximum deviation from zero of this running sum.

    .. math::

        ES = L_{arg max |L|}

    When multiple random permutations are done (``times > 1``), statistical significance is assessed via empirical testing.

    .. math::

        p_{value}=\frac{ES_{rand} \geq ES}{P}

    Where:

    - :math:`ES_{rand}` are the enrichment scores of the random permutations
    - :math:`P` is the total number of permutations

    Additionaly, :math:`ES` is updated to a normalized enrichment score :math:`NES`.

    .. math::

        NES = \begin{cases} \frac{ES}{\mu_{+}} & \text{if } ES > 0 \\ \frac{ES}{\mu_{-}} & \text{if } ES < 0  \end{cases}

    Where:

    - :math:`\mu{+}` is the mean of positive values in :math:`ES_{rand}`
    - :math:`\mu{-}` is the mean of negative values in :math:`ES_{rand}`

    %(yestest)s

    %(params)s
    %(times)s
    %(seed)s

    %(returns)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc
        adata, net = dc.ds.toy()
        dc.mt.gsea(adata, net, tmin=3)
    """

    nobs, nvar = mat.shape
    assert isinstance(times, int | float) and times >= 0, "times must be numeric and >= 0"
    assert isinstance(seed, int | float) and seed >= 0, "seed must be numeric and >= 0"
    times, seed = int(times), int(seed)

    # Compute
    nsrc = starts.size    
    m = f"gsea - calculating {nsrc} scores across {nobs} observations"
    _log(m, level="info", verbose=verbose)
    if times > 1:
        m = f"gsea - comparing estimates against {times} random permutations"
        _log(m, level="info", verbose=verbose)
        ridx = _ridx(times=times, nvar=nvar, seed=0)
    else:
        ridx = _ridx(times=times, nvar=nvar, seed=None)
    
    es, nes, pv = _stsgsea(mat, cnct, starts, offsets, ridx, times, seed, verbose)
    if times > 1:
        es = nes   
    return es.get(), pv.get()


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
