import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
def predictions(algorithm, graph, context, seed, *, factors, iterations, regularization, svd_iterations, hybrid_weight):
    if not 1 <= factors < min(graph.shape[0]+context.shape[0], context.shape[1]):
        raise ValueError("Factor count must be positive and smaller than the matrix dimensions")
    if iterations < 1 or svd_iterations < 1 or regularization < 0 or not 0 <= hybrid_weight <= 1:
        raise ValueError("Invalid training settings")
    random.seed(seed)
    np.random.seed(seed)
    matrix = sp.vstack([graph, context], format="csr").astype(np.float64)
    if algorithm == "PureSVD":
        u, singular, vt = svds(matrix, k=factors, random_state=seed)
        score = (u[-context.shape[0]:] * singular) @ vt
    elif algorithm == "DR-MF":
        model = TruncatedSVD(n_components=factors, n_iter=svd_iterations, random_state=seed)
        latent = model.fit_transform(matrix)
        score = model.inverse_transform(latent[-context.shape[0]:])
    elif algorithm == "ALS":
        model = AlternatingLeastSquares(
            factors=factors, iterations=iterations, regularization=regularization,
            random_state=seed, num_threads=4,
        )
        model.fit(matrix, show_progress=False)
        score = model.user_factors[-context.shape[0]:] @ model.item_factors.T
    elif algorithm == "HybridRec":
        als = AlternatingLeastSquares(
            factors=factors, iterations=iterations, regularization=regularization,
            random_state=seed, num_threads=4,
        )
        als.fit(matrix, show_progress=False)
        als_score = als.user_factors[-context.shape[0]:] @ als.item_factors.T
        svd = TruncatedSVD(n_components=factors, n_iter=svd_iterations, random_state=seed)
        latent = svd.fit_transform(matrix)
        svd_score = svd.inverse_transform(latent[-context.shape[0]:])
        score = hybrid_weight * als_score + (1-hybrid_weight) * svd_score
    else:
        raise ValueError(algorithm)
    score[context.nonzero()] = -np.inf
    return score
