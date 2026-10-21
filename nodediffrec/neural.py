import random
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import TruncatedSVD
from .ncf import NCF
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
class GraphCF(nn.Module):
    """Architecture used by the repository's NGCF/SGL/SimGCL implementations."""

    def __init__(self, n_users, n_items, norm_adj, embed_size=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.register_buffer("norm_adj", norm_adj)
        self.embedding_user = nn.Embedding(n_users, embed_size)
        self.embedding_item = nn.Embedding(n_items, embed_size)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        self.W_gc = nn.ModuleList(nn.Linear(embed_size, embed_size) for _ in range(n_layers))
        self.W_bi = nn.ModuleList(nn.Linear(embed_size, embed_size) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    def embeddings(self, adj=None):
        if adj is None:
            adj = self.norm_adj
        ego = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        layers = [ego]
        for layer in range(self.n_layers):
            side = torch.sparse.mm(adj, ego)
            ego = F.leaky_relu(self.W_gc[layer](side) + self.W_bi[layer](ego * side))
            ego = self.dropout(ego)
            layers.append(ego)
        return torch.cat(layers, dim=1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalized_adjacency(matrix):
    n_users, n_items = matrix.shape
    rows, cols = matrix.nonzero()
    user_nodes = rows
    item_nodes = cols + n_users
    a_rows = np.concatenate([user_nodes, item_nodes])
    a_cols = np.concatenate([item_nodes, user_nodes])
    degree = np.bincount(a_rows, minlength=n_users + n_items).astype(np.float32)
    inv = np.zeros_like(degree)
    positive = degree > 0
    inv[positive] = degree[positive] ** -0.5
    values = inv[a_rows] * inv[a_cols]
    indices = torch.tensor(np.vstack([a_rows, a_cols]), dtype=torch.long, device=DEVICE)
    values = torch.tensor(values, dtype=torch.float32, device=DEVICE)
    return torch.sparse_coo_tensor(
        indices, values, (n_users + n_items, n_users + n_items), device=DEVICE
    ).coalesce()

def drop_edge(adj, probability=0.2):
    values = adj.values()
    keep = torch.rand(values.shape, device=values.device) >= probability
    scaled = values[keep] / (1.0 - probability)
    return torch.sparse_coo_tensor(adj.indices()[:, keep], scaled, adj.shape, device=DEVICE).coalesce()

def svd_initialization(matrix, rank):
    rank = min(rank, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=rank, n_iter=50, random_state=0)
    users = svd.fit_transform(matrix).astype(np.float32)
    items = svd.components_.T.astype(np.float32)
    return users, items

def sample_negatives(matrix, users, rng):
    negatives = rng.integers(0, matrix.shape[1], size=len(users), dtype=np.int64)
    for index, user in enumerate(users):
        start, end = matrix.indptr[user], matrix.indptr[user + 1]
        positives = matrix.indices[start:end]
        while np.any(positives == negatives[index]):
            negatives[index] = rng.integers(0, matrix.shape[1])
    return negatives

def info_nce(first, second, temperature=0.2):
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return F.cross_entropy(logits, labels)

def score_model(model, algorithm, eval_user_start, n_eval_users, n_items):
    model.eval()
    if algorithm == "NeuMF":
        rows = []
        items = torch.arange(n_items, dtype=torch.long, device=DEVICE)
        for offset in range(n_eval_users):
            users = torch.full((n_items,), eval_user_start + offset, dtype=torch.long, device=DEVICE)
            rows.append(model(users, items).detach().cpu().numpy())
        return np.vstack(rows)
    embeddings = model.embeddings()
    users = embeddings[eval_user_start:eval_user_start + n_eval_users]
    items = embeddings[model.n_users:model.n_users + n_items]
    return (users @ items.T).detach().cpu().numpy()
