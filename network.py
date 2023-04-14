import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 576)
        self.fc2 = nn.Linear(576, 576)
        self.fc3 = nn.Linear(576, 1)


    def forward(self, x):
        pred = F.relu(self.fc1(x))
        pred = F.relu(self.fc2(pred))
        pred = F.sigmoid(self.fc3(pred))
        return pred
