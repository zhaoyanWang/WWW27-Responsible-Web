import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import LGConv
class LightGCNLayer(nn.Module):
    def __init__(self, normalize=True):
        super(LightGCNLayer, self).__init__()
        self.conv = LGConv(normalize=normalize)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class node_encoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64):
        super(node_encoder, self).__init__()
        self.num_layers = 2
        self.feat_encode = nn.Linear(feat_dim, hidden_dim)
        self.neighbor_encode = nn.ModuleList([LightGCNLayer() for _ in range(self.num_layers)])
        self.latent_encode = nn.Linear(hidden_dim*2, reparam_dim)
        self.mean = nn.Linear(reparam_dim, latent_dim)
        self.log_var = nn.Linear(reparam_dim, latent_dim)
        self.linear_proj = torch.nn.Linear(512,256)

    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(log_var)
        z = mean + eps * torch.exp(log_var * 0.5)
        return z
    
    def forward(self, x, edge_index, pos_emb):
        feat = F.relu(self.feat_encode(x))
        pos_emb = self.linear_proj(pos_emb)
        neighbor_feat = feat + pos_emb
        for layer in self.neighbor_encode:
            neighbor_feat = layer(neighbor_feat, edge_index)

        feat = torch.cat([feat, neighbor_feat], dim=1)
        feat = F.relu(self.latent_encode(feat))
        mean = self.mean(feat)
        log_var = self.log_var(feat)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var

class node_decoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64, neighbor_map_dim=2708):
        super(node_decoder, self).__init__()
        self.latent_decode = nn.Linear(latent_dim, reparam_dim)
        self.reparam_decode = nn.Linear(reparam_dim, hidden_dim*2)
        self.feat_decode = nn.Linear(hidden_dim, feat_dim)
        self.neighbor_decode = nn.Linear(hidden_dim, neighbor_map_dim)

    def forward(self, z, temp=0.5):
        z_decode_1 = F.relu(self.latent_decode(z))
        z_decode_2 = F.relu(self.reparam_decode(z_decode_1))
        z_decode = torch.chunk(z_decode_2, 2, dim=-1)
        feat = z_decode[0]
        neighbor_feat = z_decode[1]
        feat = self.feat_decode(feat)
        neighbor_feat = self.neighbor_decode(neighbor_feat)
        feat = torch.sigmoid(feat)
        neighbor_map = torch.sigmoid(neighbor_feat/temp)
        return feat, neighbor_map

class node_vae(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, reparam_dim=128, latent_dim=64, neighbor_map_dim=2708):
        super(node_vae, self).__init__()
        self.encoder = node_encoder(feat_dim, hidden_dim, reparam_dim, latent_dim)
        self.decoder = node_decoder(feat_dim, hidden_dim, reparam_dim, latent_dim, neighbor_map_dim=neighbor_map_dim)

    def forward(self, feat, edge_index, pos_emb):
        z, mean, log_var = self.encoder(feat, edge_index, pos_emb)
        feat, neighbor_map = self.decoder(z)
        return feat, neighbor_map, mean, log_var, z
