import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from torch_scatter import scatter_add
from utils.math_utils import artanh, tanh

try:
    import numba
except ImportError:
    numba = None


def normalize_adj_dense_semantic_Small(edge_index, x, num_nodes):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    adj_sq = torch.mm(adj, adj)
    adj_1hop = (adj > 0).float() + torch.eye(num_nodes, device=adj.device)
    deg_1 = adj_1hop.sum(dim=1); d_inv_1 = deg_1.pow(-0.5); d_inv_1[d_inv_1 == float("inf")] = 0
    norm_adj_1 = torch.mm(torch.mm(torch.diag(d_inv_1), adj_1hop), torch.diag(d_inv_1))
    adj_2hop_pure = torch.relu((adj_sq > 0).float() - adj - torch.eye(num_nodes, device=adj.device))
    x_norm = F.normalize(x, p=2, dim=-1)
    sim = torch.mm(x_norm, x_norm.t())
    sim_sharpened = torch.pow(torch.relu(sim), 2)
    adj_2hop_semantic = adj_2hop_pure * sim_sharpened + torch.eye(num_nodes, device=adj.device)
    deg_2 = adj_2hop_semantic.sum(dim=1); d_inv_2 = deg_2.pow(-0.5); d_inv_2[d_inv_2 == float("inf")] = 0
    norm_adj_2 = torch.mm(torch.mm(torch.diag(d_inv_2), adj_2hop_semantic), torch.diag(d_inv_2))
    return norm_adj_1, norm_adj_2

class HyperbolicConvDense_Small(nn.Module):
    def __init__(self, in_features, out_features, c):
        super().__init__(); self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.res_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.hop_weights = nn.Parameter(torch.tensor([0.6, 0.4]))
        nn.init.xavier_uniform_(self.weight); nn.init.xavier_uniform_(self.res_weight); nn.init.constant_(self.bias, 0)
    def forward(self, x_hyp_in, adj_1, adj_2):
        sqrt_c = torch.sqrt(self.c)
        x_tan = artanh(sqrt_c * x_hyp_in.clamp(min=-1 + 1e-7, max=1 - 1e-7)) / sqrt_c
        w = F.softmax(self.hop_weights, dim=0)
        h = torch.mm(w[0]*adj_1 + w[1]*adj_2, x_tan)
        out = torch.mm(h, self.weight.t()) + torch.mm(x_tan, self.res_weight.t()) + self.bias
        return tanh(sqrt_c * out.clamp(min=-10, max=10)) / sqrt_c

class HyperbolicConvSparse_Small(nn.Module):
    def __init__(self, in_features, out_features, c):
        super().__init__(); self.c = c; self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)); self.layer_norm = nn.LayerNorm(in_features) 
        nn.init.xavier_uniform_(self.weight); nn.init.constant_(self.bias, 0)
    def forward(self, x_hyp_in, edge_index):
        num_nodes = x_hyp_in.size(0); edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes); row, col = edge_index
        deg = degree(col, num_nodes, dtype=x_hyp_in.dtype); norm = deg.pow(-0.5)[row] * deg.pow(-0.5)[col]
        sqrt_c = torch.sqrt(self.c); x_tan = self.layer_norm(artanh(sqrt_c * x_hyp_in.clamp(min=-1+1e-6, max=1-1e-6)) / sqrt_c)
        h = scatter_add(x_tan[row] * norm.view(-1, 1), col, dim=0, dim_size=num_nodes)
        h = torch.mm(h, self.weight.t()) + self.bias
        h_hyp = tanh(sqrt_c * h) / sqrt_c
        maxnorm = 0.95 / sqrt_c; h_n = h_hyp.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return torch.where(h_n > maxnorm, h_hyp / h_n * maxnorm, h_hyp)



class DySATStructuralBlock_Large(nn.Module):
    def __init__(self, in_f, out_f, heads=4, dropout=0.2):
        super().__init__(); head_dim = out_f // heads; self.dropout = dropout
        self.gat1 = GATConv(in_f, head_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(out_f, head_dim, heads=heads, concat=True, dropout=dropout); self.act = nn.ELU()
    def forward(self, x, edge_index):
        h1 = self.act(self.gat1(F.dropout(x, p=self.dropout, training=self.training), edge_index))
        h2 = self.act(self.gat2(F.dropout(h1, p=self.dropout, training=self.training), edge_index))
        return h2

class HyperbolicConvSparse_Large(nn.Module):
    def __init__(self, in_f, out_f, c):
        super().__init__(); self.c = c; self.weight = nn.Parameter(torch.Tensor(out_f, in_f))
        self.bias = nn.Parameter(torch.Tensor(out_f)); self.layer_norm = nn.LayerNorm(in_f); self.hop_decay = nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_uniform_(self.weight); nn.init.constant_(self.bias, 0)
    def forward(self, x_hyp_in, edge_index):
        num_nodes = x_hyp_in.size(0); row, col = edge_index
        deg = degree(col, num_nodes, dtype=x_hyp_in.dtype); norm = (deg.pow(-0.5)[row] * deg.pow(-0.5)[col]).nan_to_num(0)
        sqrt_c = torch.sqrt(self.c); x_norm = x_hyp_in.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        x_tan = self.layer_norm(x_hyp_in * (artanh(sqrt_c * x_norm.clamp(max=1 - 1e-6)) / (sqrt_c * x_norm)))
        h_tan = torch.mm(x_tan, self.weight.t()) + self.bias
        h_norm_feat = F.normalize(h_tan, p=2, dim=-1); edge_sim = (h_norm_feat[row] * h_norm_feat[col]).sum(dim=-1)
        edge_weight = norm * (torch.pow(torch.relu(edge_sim), 2) + 1e-4)
        h1 = scatter_add(h_tan[row] * edge_weight.view(-1, 1), col, dim=0, dim_size=num_nodes) + h_tan
        h2 = scatter_add(h1[row] * edge_weight.view(-1, 1), col, dim=0, dim_size=num_nodes)
        decay = torch.sigmoid(self.hop_decay)
        h_out = h_tan + h1 + decay * h2; h_out_n = h_out.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        h_hyp = h_out * (tanh(sqrt_c * h_out_n) / (sqrt_c * h_out_n))
        maxnorm = 0.95 / sqrt_c; h_hyp_n = h_hyp.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return torch.where(h_hyp_n > maxnorm, h_hyp / h_hyp_n * maxnorm, h_hyp)


class GAR(nn.Module):
    def __init__(self, args, c):
        super().__init__(); self.args = args; self.c = c
        self.sparse_threshold = 2000
        self.sparse_mode = args.num_nodes > self.sparse_threshold
        self.offline_ppr_dict = {}

        if self.sparse_mode: 
            self.euc_impl = DySATStructuralBlock_Large(args.nhid, args.nhid)
            self.hyp_impl = HyperbolicConvSparse_Large(args.nhid, args.nhid, c)
            self.ln_impl = nn.LayerNorm(args.nhid)
            self.use_ppr = getattr(args, "enable_large_graph_ppr", True)
        else: 
            self.euc_impl = GCNConv(args.nhid, args.nhid)
            self.hyp_impl = HyperbolicConvDense_Small(args.nhid, args.nhid, c)
            self.use_ppr = False

        self.gate_mlp = nn.Sequential(nn.Linear(args.nhid + 1, args.nhid), nn.LeakyReLU(0.2), nn.Linear(args.nhid, 1))

    def build_offline_ppr(self, edge_index_list, num_nodes):
        if not hasattr(self, 'use_ppr') or not self.use_ppr: return
        print(f"\n[PPR Offline Build] 执行 Top-K 筛选...")
        for i, edge_index in enumerate(edge_index_list):
            key = hash(edge_index.detach().cpu().numpy().tobytes())
            if key not in self.offline_ppr_dict:
                from model.geofuse_modules import compute_ppr_topk 
                self.offline_ppr_dict[key] = compute_ppr_topk(edge_index.detach().cpu(), num_nodes).to(edge_index.device)

    def forward(self, x, edge_index, deltas):
        sqrt_c = torch.sqrt(self.c)
        if self.sparse_mode:
            
            x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
            x_hyp_in = x * (tanh(sqrt_c * x_norm) / (sqrt_c * x_norm))
            topo = self.offline_ppr_dict.get(hash(edge_index.cpu().numpy().tobytes()), edge_index) if self.use_ppr else edge_index
            x_euc = self.euc_impl(self.ln_impl(x), topo)
            x_hyp_out = self.hyp_impl(x_hyp_in, topo)
            x_hyp_out_n = x_hyp_out.norm(dim=-1, keepdim=True).clamp_min(1e-15)
            x_hyp_tan = x_hyp_out * (artanh(sqrt_c * x_hyp_out_n.clamp(max=1 - 1e-7)) / (sqrt_c * x_hyp_out_n))
        else:
            
            x_hyp_in = tanh(sqrt_c * x) / sqrt_c 
            x_euc = torch.relu(self.euc_impl(x, edge_index))
            adj_1, adj_2 = normalize_adj_dense_semantic_Small(edge_index, x, x.size(0))
            x_hyp_out = self.hyp_impl(x_hyp_in, adj_1, adj_2)
            x_hyp_tan = artanh(sqrt_c * x_hyp_out.clamp(min=-1 + 1e-7, max=1 - 1e-7)) / sqrt_c

        g = torch.sigmoid(self.gate_mlp(torch.cat([x, deltas.unsqueeze(1)], dim=-1)))
        return (1 - g) * x_euc + g * x_hyp_tan


_PPR_PUSH_NUMBA = None
def _run_ppr_push(rowptr, col, alpha, eps):
    global _PPR_PUSH_NUMBA
    def _ppr_push(rowptr, col, alpha, eps):
        num_nodes = len(rowptr) - 1; alpha_eps = alpha * eps
        js = [[0]] * num_nodes; vals = [[0.0]] * num_nodes
        for inode in range(num_nodes):
            p = {inode: 0.0}; r = {inode: alpha}; q = [inode]
            while len(q) > 0:
                unode = q.pop(); res = r[unode] if unode in r else 0.0
                if unode in p: p[unode] += res
                else: p[unode] = res
                r[unode] = 0.0; start, end = rowptr[unode], rowptr[unode + 1]
                ucount = end - start
                for vnode in col[start:end]:
                    value = (1.0 - alpha) * res / ucount
                    if vnode in r: r[vnode] += value
                    else: r[vnode] = value
                    if r[vnode] >= alpha_eps * (rowptr[vnode+1]-rowptr[vnode]):
                        if vnode not in q: q.append(vnode)
            js[inode] = list(p.keys()); vals[inode] = list(p.values())
        return js, vals
    if numba is None: raise RuntimeError("numba needed")
    if _PPR_PUSH_NUMBA is None: _PPR_PUSH_NUMBA = numba.njit(cache=False)(_ppr_push)
    return _PPR_PUSH_NUMBA(rowptr, col, alpha, eps)

def compute_ppr_topk(edge_index, num_nodes, alpha=0.25, eps=1e-4, top_k=32):
    if edge_index.numel() == 0: return edge_index.new_zeros((2, 0))
    edge_index_cpu = edge_index.detach().cpu().long(); perm = edge_index_cpu[0].argsort(); edge_index_cpu = edge_index_cpu[:, perm]
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.bincount(edge_index_cpu[0], minlength=num_nodes).cumsum(dim=0)
    cols, _ = _run_ppr_push(rowptr.numpy(), edge_index_cpu[1].numpy(), float(alpha), float(eps))
    src, tar = [], []
    for target, neighbors in enumerate(cols):
        if not neighbors: src.append(target); tar.append(target); continue
        for source in neighbors[:top_k]: src.append(target); tar.append(source)
    return torch.tensor([src, tar], dtype=torch.long, device=edge_index.device)