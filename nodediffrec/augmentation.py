import numpy as np
import scipy.sparse as sp
def minmax(a):
    return (a-a.min(1,keepdims=True))/np.maximum(np.ptp(a,axis=1,keepdims=True),1e-12)

def flags(x):
    valid=np.isfinite(x);cuts=np.quantile(x[valid],[.25,.75])
    low=valid&(x<=cuts[0]);high=valid&(x>=cuts[1])
    if np.any(low&high):high=valid&(x>cuts[0])
    return low,high,cuts

def allocate(weights,total,cap,minimum=1):
    n=len(weights);quota=np.full(n,minimum,dtype=int);left=total-n*minimum
    if left<0 or total>n*cap:raise ValueError('Impossible quota')
    while left:
        available=quota<cap;raw=weights*available;raw=raw/raw.sum()*left
        inc=np.minimum(np.floor(raw).astype(int),cap-quota)
        if inc.sum()==0:
            order=np.argsort(-raw,kind='stable');inc[order[:min(left,int(available.sum()))]]=1
        quota+=inc;left=total-int(quota.sum())
    assert quota.sum()==total and quota.max()<=cap
    return quota

def edges(scores,q):
    rows=[];cols=[]
    for u,count in enumerate(q):
        idx=np.argsort(-scores[u],kind='stable')[:count]
        rows.extend([u]*count);cols.extend(idx)
    return sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=scores.shape)
