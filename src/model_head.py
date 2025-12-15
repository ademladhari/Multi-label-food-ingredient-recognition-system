import torch
import torch.nn as nn


class SimpleHead(nn.Module):
    """Simple linear head that maps feature vectors to ingredient logits.
    Returns raw logits (no sigmoid).
    """

    def __init__(self, in_features, out_features, hidden=None, dropout=0.0):
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(in_features, out_features)
        else:
            layers = [
                nn.Linear(in_features, hidden),
                nn.ReLU(inplace=True)
            ]
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, out_features))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        # expects x shape (batch, feat_dim)
        logits = self.net(x)
        return logits
