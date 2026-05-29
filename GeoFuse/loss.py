import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from manifolds.poincare import PoincareBall

class MEL_Small(nn.Module):
    def __init__(self, args, c):
        super().__init__()
        self.manifold = PoincareBall()
        self.c = c
        self.eps = 1e-7
        self.evol_weight = float(args.evol_weight.item()) if torch.is_tensor(args.evol_weight) else float(args.evol_weight)
        self.r = nn.Parameter(torch.tensor(args.r), requires_grad=True)
        self.t = nn.Parameter(torch.tensor(args.t), requires_grad=True)

    def decode_dist(self, z, edge_index):
        z_hyp = self.manifold.proj(z, self.c)
        z_i, z_j = F.embedding(edge_index[0], z_hyp), F.embedding(edge_index[1], z_hyp)
        dist = self.manifold.sqdist(z_i, z_j, self.c).squeeze()
        probs = 1. / (torch.exp(torch.clamp((dist - self.r) / self.t, min=-20.0, max=20.0)) + 1.0)
        return torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

    def forward(self, z, target_idx, ref_idx=None, epoch=None):
        pos_p = self.decode_dist(z, target_idx)
        loss = -torch.log(pos_p).mean() if pos_p.numel() > 0 else 0.0
        neg_idx = negative_sampling(target_idx, num_nodes=z.size(0), num_neg_samples=target_idx.size(1))
        neg_p = self.decode_dist(z, neg_idx)
        loss += -torch.log(1.0 - neg_p).mean() if neg_p.numel() > 0 else 0.0
        
        l_evol, r_v, z_hyp = 0.0, self.r.abs(), self.manifold.proj(z, self.c)
        if ref_idx is not None:
            def hash_e(e): return e[0] * 1000000 + e[1]
            m_n = ~torch.isin(hash_e(target_idx), hash_e(ref_idx))
            m_r = ~torch.isin(hash_e(ref_idx), hash_e(target_idx))
            if m_n.any():
                l_evol += torch.clamp(self.manifold.sqdist(z_hyp[target_idx[0,m_n]], z_hyp[target_idx[1,m_n]], self.c).squeeze() - r_v, min=0.0).mean()
            if m_r.any():
                l_evol += torch.clamp((r_v+2.0) - self.manifold.sqdist(z_hyp[ref_idx[0,m_r]], z_hyp[ref_idx[1,m_r]], self.c).squeeze(), min=0.0).mean()
        return loss + self.evol_weight * torch.sigmoid(l_evol if isinstance(l_evol, torch.Tensor) else torch.tensor(l_evol))

class MEL_Large(nn.Module):
    def __init__(self, args, c):
        super().__init__()
        self.manifold = PoincareBall()
        self.c = c
        self.eps = 1e-7
        self.r = args.r
        self.t = args.t

    def to_hyperbolic(self, z):
        sqrt_c = torch.sqrt(self.c)
        z_n = z.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return self.manifold.proj(z * (torch.tanh(sqrt_c * z_n) / (sqrt_c * z_n)), self.c)

    def hard_negative_mining(self, z_hyp, pos_idx, num):
        
        neg = negative_sampling(pos_idx, num_nodes=z_hyp.size(0), num_neg_samples=num * 20)
        with torch.no_grad():
            dists = self.manifold.sqdist(z_hyp[neg[0]], z_hyp[neg[1]], self.c).squeeze()
            _, hard = torch.topk(dists, k=num, largest=False)
        return neg[:, hard]

    def decode_dist(self, z_hyp, edge_index):
        if edge_index is None or edge_index.numel() == 0: return torch.tensor([], device=z_hyp.device)
        z_i, z_j = F.embedding(edge_index[0], z_hyp), F.embedding(edge_index[1], z_hyp)
        dist = self.manifold.sqdist(z_i, z_j, self.c).squeeze()
        probs = 1. / (torch.exp(torch.clamp((dist - self.r) / self.t, min=-15.0, max=15.0)) + 1.0)
        return torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

    def forward(self, z, target_idx, ref_idx=None, epoch=None):
        z_hyp = self.to_hyperbolic(z)
        pos_p = self.decode_dist(z_hyp, target_idx)
        loss = -torch.log(pos_p).mean() if pos_p.numel() > 0 else 0.0
        
        num_pos = target_idx.size(1)
        
        if epoch is not None and epoch < 50:
            neg_idx = negative_sampling(target_idx, num_nodes=z_hyp.size(0), num_neg_samples=num_pos)
        else:
            neg_idx = self.hard_negative_mining(z_hyp, target_idx, num_pos)
            
        neg_p = self.decode_dist(z_hyp, neg_idx)
        loss += -torch.log(1.0 - neg_p).mean() if neg_p.numel() > 0 else 0.0
        return loss

class MEL(nn.Module):
    def __init__(self, args, c):
        super().__init__()
        if args.num_nodes > 2000:
            self.impl = MEL_Large(args, c)
        else:
            self.impl = MEL_Small(args, c)

    def forward(self, z, target_idx, ref_idx=None, epoch=None):
        return self.impl(z, target_idx, ref_idx, epoch)

    def predict(self, z, pos, neg):
        z_h = self.impl.to_hyperbolic(z) if hasattr(self.impl, 'to_hyperbolic') else self.impl.manifold.proj(z, self.impl.c)
        p, n = self.impl.decode_dist(z_h, pos), self.impl.decode_dist(z_h, neg)
        y = torch.cat([torch.ones(p.size(0)), torch.zeros(n.size(0))]).cpu().numpy()
        pred = torch.cat([p, n]).detach().nan_to_num(nan=0.5).cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)