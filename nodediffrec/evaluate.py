"""Binary ranking metrics; Recall uses the truncated denominator."""
import numpy as np
import scipy.sparse as sp
from .utils import NDCG_binary_at_k_batch, recall_at_k_batch

def metrics(scores, truth, ks=(5,10,20,50)):
    scores = np.asarray(scores)
    truth = sp.csr_matrix(truth) > 0
    if scores.ndim != 2 or scores.shape != truth.shape:
        raise ValueError("Scores and truth must have matching two-dimensional shapes")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Scores may contain masked -inf entries, not NaN or +inf")
    positive_count = np.asarray(truth.sum(axis=1)).ravel()
    order = np.argsort(-scores, axis=1, kind="stable")
    result = {}
    for k in ks:
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("Cutoffs must be positive integers")
        effective_k = min(k, scores.shape[1])
        top = order[:, :effective_k]
        hits = np.take_along_axis(truth.toarray(), top, axis=1)
        hits &= np.isfinite(np.take_along_axis(scores, top, axis=1))
        result[f"NDCG@{k}"] = NDCG_binary_at_k_batch(scores, truth, k)
        result[f"Recall@{k}"] = recall_at_k_batch(scores, truth, k)
        result[f"HR@{k}"] = np.where(positive_count > 0, hits.any(1).astype(float), np.nan)
    return result
