import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        task_type="classification",
    ):
        super().__init__()
        self.task_type = task_type
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)

    def loss_fn(self, logits, y):
        y = y.view(-1)
        if self.task_type == "classification":
            return nn.BCEWithLogitsLoss()(logits, y)
        return nn.MSELoss()(logits, y)

    def predict_from_logits(self, logits):
        if self.task_type == "classification":
            return torch.sigmoid(logits)
        return logits


from torch_geometric.nn import GINConv


class GINModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        task_type="classification",
    ):
        super().__init__()
        self.task_type = task_type
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)

    def loss_fn(self, logits, y):
        y = y.view(-1)
        if self.task_type == "classification":
            return nn.BCEWithLogitsLoss()(logits, y)
        return nn.MSELoss()(logits, y)

    def predict_from_logits(self, logits):
        if self.task_type == "classification":
            return torch.sigmoid(logits)
        return logits
