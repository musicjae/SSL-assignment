import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(165, 128)
        self.conv2 = GCNConv(128, 2)
        self.classifier = Linear(2, 1)

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        embeddings = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(embeddings)

        # return out, embeddings
        return F.sigmoid(out)
