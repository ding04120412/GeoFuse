import torch
import torch.nn as nn
from model.geofuse_modules import GAR
from model.asa_layer import ASA
from manifolds.poincare import PoincareBall

class GeoFuse(nn.Module):
    def __init__(self, args):
        super(GeoFuse, self).__init__()
        self.args = args
        self.manifold = PoincareBall()
        self.c = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.feat_encoder = nn.Linear(args.nfeat, args.nhid)
        self.gar = GAR(args, self.c)
        self.asa = ASA(args.nhid, args.causal_conv_depth)
        self.gru = nn.GRUCell(args.nhid, args.nhid)
        self.history = []

    def init_history(self): self.history = []

    def forward(self, x, edge_index, deltas):
        x_in = self.feat_encoder(x)
        x_spatial = self.gar(x_in, edge_index, deltas)
        h_hat = self.asa(self.history) if self.history else torch.zeros_like(x_spatial)
        
        z_tan = self.gru(x_spatial, h_hat)
        
        z_tan = torch.clamp(z_tan, min=-10.0, max=10.0)
        
        self.history.append(z_tan.detach())
        if len(self.history) > self.args.causal_conv_depth: self.history.pop(0)
            
        z_hyp = self.manifold.expmap0(z_tan, c=self.c)
        z_hyp = self.manifold.proj(z_hyp, c=self.c)
        
        z_hyp = z_hyp * (1.0 - 1e-3)
        
        return z_hyp, z_tan