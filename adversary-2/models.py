import torch
import torch.nn.functional as F
from torch import nn


class MIAFC(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, dropout=0.2):
        super(MIAFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ClsFC(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.2):
        super(ClsFC, self).__init__()
        self.fcn = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(True),
                    nn.Linear(128, output_dim)
                )

    def forward(self, x):
        cat_x = torch.cat(x, dim=1)
        x = self.fcn(cat_x)
        return x