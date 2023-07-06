import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
# From https://github.com/zcrabbit/vbpi-gnn/blob/main/gnnModels_slim.py
# Modified to take batch dimension

class IDConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args):
        return x

# added batch dimensions
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.transform = nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, x, edge_index, *args):
        node_degree = torch.sum(edge_index >= 0, dim=-1, keepdim=True, dtype=torch.float) + 1.0   # (n_batch, n_nodes, 3) => (n_batch, n_nodes, 1)
        x = self.transform(x) / node_degree**0.5   # (n_batch, n_nodes, in_ch) => (n_batch, n_nodes, out_ch)

        node_feature_padded = torch.cat((x, torch.zeros(x.shape[0], 1, self.out_channels)), dim=-2)   # (n_batch, n_nodes+1, out_ch)   # the last zeros are prepared for index -1
        n_nodes = x.shape[1]
        neigh_feature = torch.stack([padded1[edge_index1] for padded1, edge_index1 in zip(node_feature_padded, edge_index)])
        # (n_nodes+1, out_ch), (n_nodes, 3) => (n_nodes, 3, out_ch)
        node_and_neigh_feature = torch.cat((neigh_feature, x.view(x.shape[0], x.shape[1], 1, self.out_channels)), dim=-2)   # (n_batch, n_nodes, 4, out_ch)

        return torch.sum(node_and_neigh_feature, dim=-2) / node_degree**0.5   # (n_batch, n_nodes, out_ch)

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.mlp = nn.Sequential(nn.Linear(2*self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)

    def forward(self, x, edge_index, *args):
        batch_size = x.size()[0]
        _, _, num_edges = edge_index.size()

        node_feature_padded = torch.cat((x, torch.zeros(batch_size, 1, self.in_channels, dtype=x.dtype)), dim=-2)   # (n_batch, n_nodes+1, in_channels)   # the last zeros are prepared for index -1
        neigh_feature = torch.stack([padded1[edge_index1] for padded1, edge_index1 in zip(node_feature_padded, edge_index)])

        x_ = x.repeat(1, 1, num_edges).view(batch_size, -1, self.in_channels)
        node_and_neigh_feature = torch.cat((x_, neigh_feature.view(batch_size, -1, self.in_channels) - x_), dim=-1)
        output = self.mlp(node_and_neigh_feature)

        output = torch.where(edge_index.view(batch_size, -1, 1) != -1, output, torch.tensor(-math.inf, device=x.device))
        return torch.max(output.view(batch_size, -1, num_edges, self.out_channels), dim=-2)[0]

class GNNStack(nn.Module):
    gnnModels = {'gcn': GCNConv,
                 'edge': EdgeConv,
             }
    def __init__(self, in_channels, out_channels, num_layers=1, bias=True, aggr='sum', gnn_type='gcn', project=False, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_layers = num_layers

        self.gconvs = nn.ModuleList()
        self.gconvs.append(self.gnnModels[gnn_type](self.in_channels, self.out_channels, bias=bias, aggr=aggr, project=project))
        for i in range(self.num_layers-1):
            self.gconvs.append(self.gnnModels[gnn_type](self.out_channels, self.out_channels, bias=bias, aggr=aggr, project=project))


    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gconvs[i](x, edge_index)
            x = F.elu(x)

        return x

class MeanStdPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)

        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(self.out_channels, 2, bias=bias),)


    def forward(self, x, parent_index):
        mean_std = self.net(x)   # (n_batch, n_nodes, in_ch) -> (n_batch, n_nodes, out_ch)
        mean_std = torch.max(
            mean_std[:, :-1, :], 
            torch.stack([mean_std[batch, pa, :] for batch, pa in enumerate(parent_index)]))
            # (n_batch, n_branch, out_ch)
        mean_std = self.readout(mean_std)   # -> (n_batch, n_branch, 2)
        return mean_std[..., 0], mean_std[..., 1]

    def init_last_layer(self):
        def init_weights(m):  # zero init
            logging.debug('init %s', m)
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.fill_(0.)
            elif isinstance(m, (nn.Sequential, nn.ModuleList)):
                for m1 in m:
                    init_weights(m1)
                #m.apply(init_weights)

        init_weights(self.net)
        init_weights(self.readout)

        last_layer = self.readout[-1]
        torch.nn.init.zeros_(last_layer.weight)
        last_layer.bias.data.fill_(0.)
