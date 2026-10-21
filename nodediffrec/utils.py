import numpy as np
import torch,math
import bottleneck as bn
from scipy.sparse import csr_matrix
def mask_training_examples(sparse_training_set, dense_matrix):
    """Excludes the training examples from being tested"""
    dense_matrix[
        (sparse_training_set > 0).nonzero()] = -np.inf  # Sets the already rated values to negative inf to prevent ranking ordering
    return dense_matrix

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    """Binary NDCG; zero-positive users return NaN and are excluded explicitly.

    Stored zeros are not relevant. Ties use ascending item-column order.
    Masked (-inf) candidates cannot score hits, even when fewer than k remain.
    """
    scores = np.asarray(X_pred)
    truth = csr_matrix(heldout_batch) > 0
    if scores.shape != truth.shape or scores.ndim != 2:
        raise ValueError('Scores and labels must have the same two-dimensional shape')
    if k < 1 or np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('k must be positive; scores may contain -inf, not NaN/+inf')
    k = min(int(k), scores.shape[1])
    order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    selected = truth[rows, order]
    selected = selected.toarray() if hasattr(selected, 'toarray') else np.asarray(selected)
    hits = selected * np.isfinite(scores[rows, order])
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    counts = np.asarray(truth.sum(axis=1)).ravel().astype(int)
    idcg = np.r_[0.0, np.cumsum(discounts)][np.minimum(counts, k)]
    return np.divide(dcg, idcg, out=np.full(scores.shape[0], np.nan), where=idcg > 0)

def recall_at_k_batch(X_pred, heldout_batch, k=100):
    """Truncated Recall@K = hits / min(K, positive target count).

    Zero-positive users return NaN; masked candidates cannot score hits.
    """
    scores = np.asarray(X_pred)
    truth = csr_matrix(heldout_batch) > 0
    if scores.shape != truth.shape or k < 1:
        raise ValueError('Invalid shapes or k')
    k = min(int(k), scores.shape[1])
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    selected = truth[rows, order]
    selected = selected.toarray() if hasattr(selected, 'toarray') else np.asarray(selected)
    hits = (selected * np.isfinite(scores[rows, order])).sum(axis=1)
    denom = np.minimum(k, np.asarray(truth.sum(axis=1)).ravel())
    return np.divide(hits, denom, out=np.full(scores.shape[0], np.nan), where=denom > 0)

def split_train_test_proportion_from_csr_matrix(csr_data, test_prop=0.2,
        batch_size=None, random_seed=None, ignore_zeros=False):
    """Split stored records without mutating input or renumbering users.

    Preserve labels (including explicit zero records when ignore_zeros=False).
    Rows with fewer than two eligible records stay entirely in context.
    batch_size is retained for API compatibility; construction stays sparse.
    Use ignore_zeros=True for positive-only implicit recommendation protocols.
    """
    if not 0 < test_prop < 1:
        raise ValueError('test_prop must lie strictly between zero and one')
    data = csr_matrix(csr_data, copy=True)
    data.sum_duplicates()
    if ignore_zeros:
        data.eliminate_zeros()
    rng = np.random.default_rng(random_seed)
    selected = np.zeros(data.nnz, dtype=bool)
    for row in range(data.shape[0]):
        lo, hi = data.indptr[row:row+2]
        n = hi - lo
        if n >= 2:
            size = min(n - 1, math.ceil(test_prop * n))
            selected[lo + rng.choice(n, size=size, replace=False)] = True
    rows = np.repeat(np.arange(data.shape[0]), np.diff(data.indptr))
    def build(mask):
        return csr_matrix((data.data[mask], (rows[mask], data.indices[mask])), shape=data.shape)
    return build(~selected), build(selected)

def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return  betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps:int, alpha_bar, max_beta=0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
