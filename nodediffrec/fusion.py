import numpy as np
from .groups import norm

def standardize(scores, context):
    scores=np.asarray(scores).copy();scores[context.nonzero()]=-np.inf
    z=norm(scores);z[context.nonzero()]=0
    return z

def fuse(scores, cells, weights, context, standardized=False):
    scores=np.asarray(scores);cells=np.asarray(cells,dtype=int);weights=np.asarray(weights)
    if scores.shape[0]!=3 or weights.shape!=(9,3):raise ValueError("Expected three branches and nine group cells")
    if scores.shape[1:]!=context.shape or cells.shape!=(context.shape[0],):raise ValueError("Shape mismatch")
    if (cells<0).any() or (cells>8).any() or not np.isfinite(weights).all():raise ValueError("Invalid cells/weights")
    if not np.allclose(weights.sum(1),1,atol=1e-6):raise ValueError("Weights must sum to one; negative weights are allowed")
    x=scores if standardized else np.asarray([standardize(s,context) for s in scores])
    out=np.einsum("ub,bui->ui",weights[cells],x).astype(np.float32)
    out[context.nonzero()]=-np.inf
    return out
