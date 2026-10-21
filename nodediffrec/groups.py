import numpy as np
import scipy.sparse as sp
def split(pool,rep):
    rng=np.random.RandomState(7300+rep);hold=np.zeros(pool.nnz,bool)
    for u in range(pool.shape[0]):
        l,r=pool.indptr[u:u+2];n=r-l
        if n>=2:hold[rng.choice(np.arange(l,r),min(n-1,max(1,int(np.ceil(.2*n)))),replace=False)]=True
    rows=np.repeat(np.arange(pool.shape[0]),np.diff(pool.indptr))
    t=sp.csr_matrix((pool.data[hold],(rows[hold],pool.indices[hold])),shape=pool.shape)
    return pool-t,t

def norm(s):
    out=np.full_like(s,-1e9,dtype=np.float32)
    for i,row in enumerate(s):
        f=np.isfinite(row);v=row[f];sd=v.std();out[i,f]=(v-v.mean())/(sd if sd>1e-12 else 1.)
    return out
