import numpy as np
import torch
from torch.nn import functional as F
from . import neural as nr
from .ncf import NCF
def weighted_adjacency(matrix):
    nu, ni = matrix.shape; rows, cols = matrix.nonzero(); weights = matrix.data.astype(np.float32)
    ar = np.concatenate([rows, cols + nu]); ac = np.concatenate([cols + nu, rows]); av = np.concatenate([weights, weights])
    degree = np.bincount(ar, weights=av, minlength=nu + ni).astype(np.float32)
    inv = np.zeros_like(degree); mask = degree > 0; inv[mask] = degree[mask] ** -.5
    values = av * inv[ar] * inv[ac]
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([ar, ac]), dtype=torch.long, device=nr.DEVICE),
        torch.tensor(values, dtype=torch.float32, device=nr.DEVICE), (nu + ni, nu + ni), device=nr.DEVICE).coalesce()

def positives(matrix):
    rows, cols = matrix.nonzero()
    return rows, cols, matrix.data.astype(np.float32)

def train_neumf(matrix, seed, *, epochs, lr, batch_size):
    if epochs < 1 or lr <= 0 or batch_size < 1:
        raise ValueError("Epochs, learning rate and batch size must be positive")
    nr.set_seed(seed); nu, ni = matrix.shape; ui, ii = nr.svd_initialization(matrix, 20)
    gmf = NCF(nu, ni, factor_num=20, num_layers=3, dropout=.5, model="GMF")
    mlp = NCF(nu, ni, factor_num=20, num_layers=3, dropout=.5, model="MLP")
    with torch.no_grad():
        gmf.embed_user_GMF.weight.copy_(torch.from_numpy(ui)); gmf.embed_item_GMF.weight.copy_(torch.from_numpy(ii))
        mlp.embed_user_MLP.weight.copy_(torch.from_numpy(np.tile(ui, (1, 4)))); mlp.embed_item_MLP.weight.copy_(torch.from_numpy(np.tile(ii, (1, 4))))
    model = NCF(nu, ni, factor_num=20, num_layers=3, dropout=.5, model="NeuMF-pre", GMF_model=gmf, MLP_model=mlp).to(nr.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr); rows, cols, weights = positives(matrix); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(rows)); negatives = nr.sample_negatives(matrix, rows, rng); model.train()
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]; u=torch.as_tensor(rows[ix],dtype=torch.long,device=nr.DEVICE); p=torch.as_tensor(cols[ix],dtype=torch.long,device=nr.DEVICE); n=torch.as_tensor(negatives[ix],dtype=torch.long,device=nr.DEVICE); w=torch.as_tensor(weights[ix],device=nr.DEVICE)
            opt.zero_grad(set_to_none=True); ps=model(u,p); ns=model(u,n)
            loss=(F.binary_cross_entropy_with_logits(ps,torch.ones_like(ps),reduction='none')*w).sum()/w.sum()
            loss+=F.binary_cross_entropy_with_logits(ns,torch.zeros_like(ns)); loss.backward(); opt.step()
    return model

def train_graph(matrix, algorithm, seed, *, epochs, lr, batch_size):
    if epochs < 1 or lr <= 0 or batch_size < 1:
        raise ValueError("Epochs, learning rate and batch size must be positive")
    nr.set_seed(seed); nu, ni = matrix.shape; adj = weighted_adjacency(matrix); model = nr.GraphCF(nu, ni, adj).to(nr.DEVICE)
    ui, ii = nr.svd_initialization(matrix, 64)
    with torch.no_grad(): model.embedding_user.weight.copy_(torch.from_numpy(ui).to(nr.DEVICE)); model.embedding_item.weight.copy_(torch.from_numpy(ii).to(nr.DEVICE))
    opt=torch.optim.Adam(model.parameters(),lr=lr);rows,cols,weights=positives(matrix);rng=np.random.default_rng(seed)
    for _ in range(epochs):
        order=rng.permutation(len(rows));neg=nr.sample_negatives(matrix,rows,rng);model.train()
        for start in range(0,len(order),batch_size):
            ix=order[start:start+batch_size];u=torch.as_tensor(rows[ix],dtype=torch.long,device=nr.DEVICE);p=torch.as_tensor(cols[ix],dtype=torch.long,device=nr.DEVICE);n=torch.as_tensor(neg[ix],dtype=torch.long,device=nr.DEVICE);w=torch.as_tensor(weights[ix],device=nr.DEVICE)
            opt.zero_grad(set_to_none=True); emb=model.embeddings(); ps=(emb[u]*emb[nu+p]).sum(1); ns=(emb[u]*emb[nu+n]).sum(1); per=-F.logsigmoid(ps-ns);loss=(per*w).sum()/w.sum()
            nodes=torch.unique(torch.cat([u,nu+p,nu+n]))
            if algorithm=="NGSGL": loss=loss+1e-6*nr.info_nce(model.embeddings(nr.drop_edge(adj))[nodes],model.embeddings(nr.drop_edge(adj))[nodes])
            elif algorithm=="NGSimGCL":
                chosen=emb[nodes]; perturbed=chosen+.1*torch.sign(chosen)*F.normalize(chosen,dim=1);loss=loss+1e-6*nr.info_nce(perturbed,chosen)
            loss.backward();opt.step()
    return model
