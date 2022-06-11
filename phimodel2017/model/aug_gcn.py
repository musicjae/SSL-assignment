import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F


class AUGGCN(torch.nn.Module):
    def __init__(self):
        super(AUGGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(165, 128)
        self.conv2 = GCNConv(128, 2)
        self.classifier = Linear(2, 1)
        self.dropout= torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        embeddings = self.relu(h)  # Final GNN embedding space.
        embeddings=self.dropout(embeddings)
        # Apply a final (linear) classifier.
        out = self.classifier(embeddings)

        # return out, embeddings
        return F.sigmoid(out)