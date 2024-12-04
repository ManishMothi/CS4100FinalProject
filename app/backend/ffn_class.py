import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


'''
Feed Forward Neural Network architecture.
'''
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(13, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 8)
    
    def forward(self, x):
        # x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
