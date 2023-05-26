import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Linear(56*56, 576)
        self.fc1 = nn.Linear(576, 128)
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
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 8)
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
        self.conv2 = nn.Conv1d(32, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1) # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0] # Global max pooling

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class VoxelCNN(nn.Module):
    def __init__(self, num_classes):
        super(VoxelCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(72, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, kernel_size=(6, 6, 8)) # Cx6x6x1
        x = x.view(-1, 72)

        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    


#-------------------------------------------------------------
# Models for Milestone

class VanillaPointNet1(torch.nn.Module):
    """Vanilla PointNet model"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 128, 1)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1) # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0] # Global max pooling

        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    

class VoxelCNN1(nn.Module):
    def __init__(self, num_classes):
        super(VoxelCNN1, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(72, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, kernel_size=(6, 6, 8)) # Cx6x6x1
        x = x.view(-1, 72)

        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    