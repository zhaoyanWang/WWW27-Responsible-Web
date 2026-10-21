"""Node generation using training-only validation and conditional diffusion."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import from_scipy_sparse_matrix
from .gg_models import node_vae
from .position import create_positional_embeddings
from .unet_1d import Unet_1d
from .embedding import ConditionalEmbedding
from .diffusion import GaussianDiffusion
from .utils import get_named_beta_schedule


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def decode(model, z):
    d = model.decoder
    h = F.relu(d.reparam_decode(F.relu(d.latent_decode(z))))
    f, a = h.chunk(2, dim=-1)
    return torch.sigmoid(d.feat_decode(f)), d.neighbor_decode(a) / 0.5


class Stopper:
    def __init__(self, out, name, patience, minimum, maximize=False):
        self.out, self.name = out, name
        self.patience, self.minimum, self.maximize = patience, minimum, maximize
        self.best = -float('inf') if maximize else float('inf')
        self.epoch, self.bad = 0, 0
        self.start = time.time()
        self.records = []

    def step(self, epoch, train, val, state):
        if not np.isfinite(train + val):
            raise RuntimeError('Nonfinite loss in ' + self.name)
        better = val > self.best + 1e-6 if self.maximize else val < self.best - 1e-6
        if better:
            self.best, self.epoch, self.bad = val, epoch, 0
            torch.save(state, self.out / (self.name + '_best.pt'))
        else:
            self.bad += 1
        row = dict(epoch=epoch, train_loss=train, validation=val,
                   best_epoch=self.epoch, elapsed_seconds=time.time()-self.start)
        self.records.append(row)
        dump(self.out / (self.name + '_history.json'), self.records)
        print(self.name, json.dumps(row), flush=True)
        return epoch >= self.minimum and self.bad >= self.patience

    def finish(self, epoch, maximum):
        return dict(best_epoch=self.epoch, executed_epochs=epoch,
                    best_validation=self.best, seconds=time.time()-self.start,
                    stopped_by='patience' if epoch < maximum else 'safety_cap')


def train(args):
    """GG training with caller-supplied settings; no experiment preset.

    Required fields: train/output Paths, device, seed, gcn_max, feature_max,
    graph_max, diffusion_max, new_items, new_edges, timesteps, gcn_lr,
    feature_lr, graph_lr, diffusion_lr, diffusion_weight_decay, guidance,
    variance_interpolation, batch_size, grad_clip, patience, minimum_epochs,
    check_every.
    """
    if args.check_every < 1 or min(args.gcn_max, args.feature_max, args.graph_max, args.diffusion_max) < args.check_every:
        raise ValueError("Each training budget must include at least one validation check")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Use a new output directory; existing runs are immutable.')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    source = args.train
    full = (sp.load_npz(source)>0).astype(np.float32).tocsr()
    nu, ni = full.shape
    n = nu + ni
    # Withhold approximately 10% (at least one candidate) per eligible user,
    # retaining at least one observed edge per user and item.
    counts = np.asarray(full.sum(0)).ravel().astype(int)
    vr, vc = [], []
    for u in rng.permutation(nu):
        ids = full[u].indices
        if len(ids) < 3:
            continue
        for i in rng.permutation(ids)[:max(1, int(len(ids)*0.1))]:
            if counts[i] > 1:
                vr.append(u); vc.append(i); counts[i] -= 1
    val = sp.csr_matrix((np.ones(len(vr), dtype=np.float32), (vr, vc)), shape=full.shape)
    fit = (full - val).tocsr(); fit.eliminate_zeros()
    sp.save_npz(out / 'internal_fit.npz', fit)
    sp.save_npz(out / 'internal_validation.npz', val)
    adj = sp.bmat([[None, fit], [fit.T, None]], format='csr')
    edges, _ = from_scipy_sparse_matrix(adj)
    edges = edges.to(device)
    y = torch.tensor(fit.toarray(), device=device)
    known = torch.tensor(full.toarray() > 0, device=device)
    val_mask = torch.tensor(val.toarray() > 0, device=device)
    eligible = np.flatnonzero(np.diff(val.indptr))
    truth = val.toarray() > 0
    discounts = 1 / np.log2(np.arange(2, 12))

    def ndcg(scores):
        a = scores.detach().cpu().numpy().copy()
        a[fit.toarray() > 0] = -np.inf
        order = np.argsort(-a[eligible], axis=1, kind='stable')[:, :10]
        hits = np.take_along_axis(truth[eligible], order, axis=1)
        ideal = np.array([discounts[:min(10, int(truth[u].sum()))].sum() for u in eligible])
        return float(np.mean((hits*discounts).sum(1) / ideal))

    protocol = dict(train_shape=list(full.shape),
        train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        internal_fit_edges=fit.nnz, internal_validation_edges=val.nnz,
        selection='Training-only validation; best checkpoint restored',
        evaluation_interval=args.check_every, patience=args.patience,
        minimum_epochs=args.minimum_epochs)
    dump(out / 'protocol.json', protocol)
    summary = {}
    start = time.time()

    gcn = LightGCN(n, embedding_dim=512, num_layers=2).to(device)
    opt = torch.optim.Adam(gcn.parameters(), lr=args.gcn_lr)
    coo = fit.tocoo()
    us = torch.tensor(coo.row, device=device, dtype=torch.long)
    it = torch.tensor(coo.col, device=device, dtype=torch.long)
    stop = Stopper(out, 'lightgcn', args.patience, args.minimum_epochs, maximize=True)
    for ep in range(1, args.gcn_max+1):
        gcn.train(); total = 0
        for idx in torch.randperm(len(us), device=device).split(8192):
            u, i = us[idx], it[idx]
            neg = torch.randint(ni, (len(idx),), device=device)
            bad = known[u, neg]
            while bad.any():
                neg[bad] = torch.randint(ni, (int(bad.sum()),), device=device)
                bad = known[u, neg]
            opt.zero_grad()
            z = gcn.get_embedding(edges)
            pos = (z[u]*z[nu+i]).sum(1); ns = (z[u]*z[nu+neg]).sum(1)
            loss = gcn.recommendation_loss(pos, ns, node_id=torch.cat([u, nu+i, nu+neg]).unique())
            loss.backward(); opt.step(); total += float(loss)*len(idx)/len(us)
        if ep % args.check_every == 0:
            gcn.eval()
            with torch.no_grad():
                z = gcn.get_embedding(edges)
                metric = ndcg(z[:nu] @ z[nu:].T)
            if stop.step(ep, total, metric, gcn.state_dict()): break
    summary['lightgcn'] = stop.finish(ep, args.gcn_max)
    gcn.load_state_dict(torch.load(out / 'lightgcn_best.pt'))
    with torch.no_grad(): feat = gcn.get_embedding(edges).detach()
    np.save(out / 'node_embeddings.npy', feat.cpu().numpy())
    feat = (feat-feat.min())/(feat.max()-feat.min()).clamp_min(1e-8)
    pos = create_positional_embeddings(n, 512).to(device)
    model = node_vae(neighbor_map_dim=n).to(device)
    held = np.concatenate([rng.choice(nu, max(1, nu//10), replace=False),
                           nu+rng.choice(ni, max(1, ni//10), replace=False)])
    held_mask = torch.zeros(n, device=device, dtype=torch.bool); held_mask[held] = True
    np.save(out / 'feature_validation_nodes.npy', held)
    opt = torch.optim.Adam(model.parameters(), lr=args.feature_lr)
    stop = Stopper(out, 'feature', args.patience, args.minimum_epochs)
    for ep in range(1, args.feature_max+1):
        model.train(); opt.zero_grad()
        z, _, _ = model.encoder(feat, edges, pos)
        rec, _ = decode(model, z)
        loss = F.mse_loss(rec[~held_mask], feat[~held_mask])
        loss.backward(); opt.step()
        if ep % args.check_every == 0:
            model.eval()
            with torch.no_grad():
                _, mu, _ = model.encoder(feat, edges, pos)
                rec, _ = decode(model, mu)
                metric = float(F.mse_loss(rec[held_mask], feat[held_mask]))
            if stop.step(ep, float(loss), metric, model.state_dict()): break
    summary['feature'] = stop.finish(ep, args.feature_max)
    model.load_state_dict(torch.load(out / 'feature_best.pt'))
    for param in model.encoder.parameters(): param.requires_grad_(False)
    with torch.no_grad():
        _, mu, lv = model.encoder(feat, edges, pos)
    opt = torch.optim.Adam(model.decoder.parameters(), lr=args.graph_lr)
    stop = Stopper(out, 'graph', args.patience, args.minimum_epochs, maximize=True)
    for ep in range(1, args.graph_max+1):
        model.decoder.train(); opt.zero_grad()
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        rec, logits = decode(model, z)
        a, b = logits[:nu, nu:], logits[nu:, :nu].T
        mask = ~val_mask
        loss = F.mse_loss(rec, feat) + 0.5*(F.binary_cross_entropy_with_logits(a[mask], y[mask]) + F.binary_cross_entropy_with_logits(b[mask], y[mask]))
        loss.backward(); opt.step()
        if ep % args.check_every == 0:
            model.eval()
            with torch.no_grad():
                _, logits = decode(model, mu)
                # Evaluate precisely the item-to-user branch used for injection.
                metric = ndcg(logits[nu:, :nu].T)
            if stop.step(ep, float(loss), metric, model.state_dict()): break
    summary['graph'] = stop.finish(ep, args.graph_max)
    model.load_state_dict(torch.load(out / 'graph_best.pt')); model.eval()
    latent = mu.detach()
    np.save(out / 'latents.npy', latent.cpu().numpy())
    dump(out / 'training_summary.json', summary)

    labels = torch.cat([torch.zeros(nu, dtype=torch.long), torch.ones(ni, dtype=torch.long)]).to(device)
    dv = np.concatenate([rng.choice(nu, nu//10, replace=False), nu+rng.choice(ni, ni//10, replace=False)])
    dt = np.setdiff1d(np.arange(n), dv)
    np.save(out / 'diffusion_validation_nodes.npy', dv)
    low, high = latent[dt].min(), latent[dt].max()
    x = ((latent-low)/(high-low).clamp_min(1e-8)).unsqueeze(1)
    dump(out / 'latent_scaling.json', dict(min=float(low), max=float(high)))
    net = Unet_1d(in_ch=1, mod_ch=64, out_ch=1, ch_mul=[1,2,2,2],
                  num_res_blocks=2, cdim=10, use_conv=True, droprate=0.1, dtype=torch.float32)
    emb = ConditionalEmbedding(2,10,10).to(device)
    diff = GaussianDiffusion(torch.float32, net, get_named_beta_schedule(num_diffusion_timesteps=args.timesteps), args.guidance, args.variance_interpolation, device)
    opt = torch.optim.AdamW(list(net.parameters())+list(emb.parameters()), lr=args.diffusion_lr, weight_decay=args.diffusion_weight_decay)
    # Fixed noise and timesteps eliminate Monte Carlo changes between checks.
    xv = x[dv].repeat(4,1,1); lv_cond = labels[dv].repeat(4)
    vg = torch.Generator(device=device).manual_seed(args.seed+10)
    tv = torch.randint(args.timesteps, (len(xv),), generator=vg, device=device)
    noise = torch.randn(xv.shape, generator=vg, device=device)
    noisy = diff._extract(diff.sqrt_alphas_bar, tv, xv.shape)*xv + diff._extract(diff.sqrt_one_minus_alphas_bar, tv, xv.shape)*noise
    stop = Stopper(out, 'diffusion', args.patience, args.minimum_epochs)
    for ep in range(1, args.diffusion_max+1):
        net.train(); emb.train(); total=0
        for idx in torch.tensor(rng.permutation(dt), device=device).split(args.batch_size):
            opt.zero_grad(); c = emb(labels[idx])
            c = c * (torch.rand(len(idx),1,device=device)>=0.1)
            loss = diff.trainloss(x[idx], cemb=c)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step(); total += float(loss)*len(idx)/len(dt)
        if ep % args.check_every == 0:
            net.eval(); emb.eval(); metric=0
            with torch.no_grad():
                for ids in torch.arange(len(xv),device=device).split(args.batch_size):
                    metric += float(F.mse_loss(net(noisy[ids], tv[ids], cemb=emb(lv_cond[ids])), noise[ids]))*len(ids)/len(xv)
            if stop.step(ep, total, metric, dict(net=net.state_dict(), embedding=emb.state_dict())): break
    summary['diffusion'] = stop.finish(ep, args.diffusion_max)
    ck = torch.load(out / 'diffusion_best.pt')
    net.load_state_dict(ck['net']); emb.load_state_dict(ck['embedding'])
    net.eval(); emb.eval()
    torch.manual_seed(args.seed+100)
    samples=[]
    with torch.no_grad():
        for offset in range(0,args.new_items,args.batch_size):
            size=min(args.batch_size,args.new_items-offset)
            cond=emb(torch.ones(size,device=device,dtype=torch.long))
            xt=torch.randn(size,1,64,device=device)
            for t in reversed(range(args.timesteps)):
                ts=torch.full((size,),t,device=device,dtype=torch.long)
                xt=diff.p_sample(xt,ts,cemb=cond)
            samples.append(xt.clamp(0,1).squeeze(1)*(high-low)+low)
            print('generated', offset+size, flush=True)
        generated=torch.cat(samples)
        _, logits=decode(model, generated)
        scores=torch.sigmoid(logits[:,:nu]).T.cpu().numpy()
    np.save(out / 'generated_latents.npy', generated.cpu().numpy())
    np.save(out / 'generated_edge_scores.npy', scores)
    # Select the requested edge budget with stable score ordering.
    idx=np.argsort(-scores.ravel(),kind='stable')[:args.new_edges]
    rr,cc=np.unravel_index(idx,scores.shape)
    aug=sp.csr_matrix((np.ones(len(idx),dtype=np.float32),(rr,cc)),shape=scores.shape)
    merged=sp.hstack([full,aug],format='csr')
    assert (merged[:,:ni]!=full).nnz==0 and merged.nnz==full.nnz+args.new_edges
    assert np.isfinite(scores).all() and generated.shape==(args.new_items,64)
    sp.save_npz(out / 'train_augmented.npz',merged)
    # Evaluation contexts can be zero-padded for auxiliary-item columns.
    # Graph generation does not read evaluation interactions.
    degrees=np.asarray(aug.sum(1)).ravel()
    summary['augmentation']=dict(shape=list(merged.shape),real_items=ni,new_items=args.new_items,
        added_edges=aug.nnz,active_new_items=int(np.count_nonzero(aug.getnnz(axis=0))),
        connected_train_users=int(np.count_nonzero(degrees)),max_user_added_degree=int(degrees.max()),
        top10_users_edge_fraction=float(np.sort(degrees)[-10:].sum()/aug.nnz),
        score_range=[float(scores.min()),float(scores.max())], selected_min_score=float(scores.ravel()[idx[-1]]),
        original_block_exact=True,final_test_read=False)
    summary['total_seconds']=time.time()-start
    dump(out / 'training_summary.json',summary)
    print('COMPLETE',json.dumps(summary),flush=True)



