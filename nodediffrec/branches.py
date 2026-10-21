"""Parameter-driven branch construction; no experiment presets."""
import numpy as np
import scipy.sparse as sp
from scipy.special import expit, logit
from scipy.stats import rankdata
from . import neural
from .augmentation import minmax, flags, allocate, edges
from .refine import conditional_decode

def build_branch(real, gg_scores, training_quality, diffusion, vae,
                 params, timesteps, noise_scale, seed):
    """Construct a GG+SD graph using externally supplied branch parameters."""
    gg = minmax(gg_scores)
    degree = np.asarray(real.sum(1)).ravel()
    q = np.asarray(training_quality)
    al = flags(degree)[0].astype(float)
    ql = flags(q)[0].astype(float)
    diff = diffusion
    b=params;aa,qq=al,ql
    if b.get("weight_form")=="continuous":
        aa=1-(rankdata(degree)-.5)/len(degree);qq=np.zeros_like(ql);valid=np.isfinite(q);qq[valid]=1-(rankdata(q[valid])-.5)/valid.sum()
    importance=np.exp(b["activity"]*aa+b["quality"]*qq+b["joint"]*aa*qq)
    quota=allocate(importance,round(b["mean_edges"]*real.shape[0]),min(gg.shape[1],int(np.ceil(b["cap_multiplier"]*b["mean_edges"]))),b.get("minimum",1))
    first=sp.hstack([real,edges(gg,quota)],format="csr")
    neural.set_seed(seed)
    decoded,_=conditional_decode(first,diff,vae,timesteps,b["fraction"],noise_scale*b["noise_multiplier"],seed)
    lam=expit(logit(b["sd_base"])+b["sd_activity"]*aa+b["sd_quality"]*qq)
    virtual=edges((1-lam[:,None])*gg+lam[:,None]*minmax(decoded[:,real.shape[1]:]),quota)
    graph=sp.hstack([real,virtual*b["edge_scale"]],format="csr")
    assert (graph[:,:real.shape[1]]!=real).nnz==0
    return graph, quota, lam
