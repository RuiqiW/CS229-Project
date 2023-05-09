import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Linear(56*56, 2048)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc(x.view(-1, 56*56))
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    


class VanillaPointNet(torch.nn.Module):
    """Vanilla PointNet model"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1) # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0] # Global max pooling
        # x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)