import numpy as np
import scipy.sparse as sp
from . import matrix, neural, weighted

MATRIX = {"ALS", "DR-MF", "PureSVD", "HybridRec"}

def predict(algorithm, graph, context, seed, **training_settings):
    if algorithm in MATRIX:
        return matrix.predictions(algorithm, graph, context, seed, **training_settings)
    x = sp.vstack([graph, context], format="csr").astype(np.float32)
    x.eliminate_zeros(); x.sort_indices()
    model = (weighted.train_neumf(x, seed, **training_settings) if algorithm == "NeuMF"
             else weighted.train_graph(x, algorithm, seed, **training_settings))
    result = neural.score_model(model, algorithm, graph.shape[0], context.shape[0], context.shape[1])
    result[context.nonzero()] = -np.inf
    return result
