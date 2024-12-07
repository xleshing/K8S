import torch
import torch.nn as nn

class FcLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
