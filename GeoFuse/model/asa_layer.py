import torch
import torch.nn as nn
import torch.nn.functional as F

class ASA(nn.Module):
    def __init__(self, input_dim, window_size, dropout=0.3):
        super(ASA, self).__init__()
        self.window_size = window_size
        self.map_linear = nn.Linear(input_dim * window_size, input_dim * window_size * 2)
        self.att_linear = nn.Linear(input_dim * window_size * 2, window_size)
        self.dropout = dropout

    def forward(self, history_list):
        if not history_list: return None
        while len(history_list) < self.window_size:
            history_list = [history_list[0]] + history_list
        
        valid_hist = history_list[-self.window_size:]
        concat_hist = torch.cat(valid_hist, dim=1) # [N, w*d]
        
        z_asa = torch.tanh(self.map_linear(concat_hist))
        z_asa = F.dropout(z_asa, self.dropout, training=self.training)
        
        att_weights = F.softmax(self.att_linear(z_asa), dim=1) # [N, w]
        stack_hist = torch.stack(valid_hist, dim=1) # [N, w, d]
        h_hat = (stack_hist * att_weights.unsqueeze(-1)).sum(dim=1)
        return h_hat