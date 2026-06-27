import torch
import torch.nn as nn


class FingerprintMLP(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        dropout=0.2,
        task_type="classification",
    ):
        super().__init__()

        self.task_type = task_type

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def loss_fn(self, logits, y):
        if self.task_type == "classification":
            return nn.BCEWithLogitsLoss()(logits, y)
        return nn.MSELoss()(logits, y)

    def predict(self, x):
        logits = self.forward(x)
        if self.task_type == "classification":
            return torch.sigmoid(logits)
        return logits
