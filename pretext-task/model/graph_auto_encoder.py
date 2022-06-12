import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F


class GrapghAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(GrapghAutoEncoder, self).__init__()
        torch.manual_seed(12345)
        self.graph_enocder = GCNConv(165, 128)
        self.graph_encoder2 = GCNConv(128, 64)
        self.decoder = Linear(64, 165)
        self.relu = torch.nn.ReLU()

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        h = self.graph_enocder(x, edge_index)
        h = self.relu(h)
        h = self.graph_encoder2(h, edge_index)
        embeddings = self.relu(h)

        out = self.decoder(embeddings)
        return out