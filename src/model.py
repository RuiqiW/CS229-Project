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
    

class MultiViewCNN(nn.Module):
    def __init__(self, num_classes, num_views=12):
        super().__init__()
        self.num_views = num_views

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, num_classes)

    
    def forward(self, x):
        batchsize = x.size()[0]
        x = x.view(-1, 56, 56)
        x = x.unsqueeze(dim=1)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, padding=1)  # 28

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 4, padding=1) # 7

        x = x.view(batchsize, self.num_views, 16, 7, 7)
        x = torch.max(x, dim=1)[0]
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
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
    "3D CNN for isotrophic (36x36x36) voxels"
    def __init__(self, num_classes):
        super(VoxelCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, kernel_size=(9, 9, 9)) # Cx4x4x4
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class VoxelCNNProbing(nn.Module):
    def __init__(self, num_classes):
        super(VoxelCNNProbing, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 4, kernel_size=(1, 1, 36))
        self.conv2 = nn.Conv3d(1, 1, kernel_size=(4, 1, 1))

        self.conv3 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(648, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = x.squeeze()
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv2(x))

        x = x.squeeze(1)
        x = F.relu(self.conv3(x)) # 2, 36, 36
        x = F.max_pool2d(x, 2) # 2, 18, 18
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VoxelCNNProbingSub(nn.Module):
    def __init__(self, axis=0):
        super(VoxelCNNProbingSub, self).__init__()
        if axis == 0:
            self.conv1 = nn.Conv3d(1, 4, kernel_size=(1, 1, 36))
        elif axis == 1:
            self.conv1 = nn.Conv3d(1, 4, kernel_size=(1, 36, 1))
        else:
            self.conv1 = nn.Conv3d(1, 4, kernel_size=(36, 1, 1))
        self.conv2 = nn.Conv3d(1, 1, kernel_size=(4, 1, 1))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.squeeze()
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv2(x))
        x = x.squeeze(1)    # (B, 1, 36, 36)

        return x
        
class VoxelCNNProbingMul(nn.Module):
    def __init__(self, num_classes):
        super(VoxelCNNProbingMul, self).__init__()
        
        # self.conv1 = nn.Conv3d(1, 4, kernel_size=(1, 1, 36))
        # self.conv2 = nn.Conv3d(1, 1, kernel_size=(4, 1, 1))

        self.nn1 = VoxelCNNProbingSub(axis=0)
        self.nn2 = VoxelCNNProbingSub(axis=1)
        self.nn3 = VoxelCNNProbingSub(axis=2)

        self.conv3 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(648, 128)
        self.fc2 = nn.Linear(128, num_classes)
    

    def forward(self, x):
        # x0 = x.unsqueeze(1)
        # x1 = x0.clone()
        # x2 = x0.clone()
        # x1 = x1.permute(0, 1, 3, 4, 2)
        # x2 = x2.permute(0, 1, 4, 2, 3)
        # x = torch.hstack([x0, x1, x2])  # (B, 3, 36, 36, 36)
        # x = x.view(-1, 36, 36, 36)  
        # x = x.unsqueeze(dim=1)  # (Bx3, 1, 36, 36, 36)

        # x = F.relu(self.conv1(x))
        # x = x.squeeze()
        # x = x.unsqueeze(dim=1)
        # x = F.relu(self.conv2(x))

        # x = x.squeeze() # (Bx3, 36, 36)
        # x = x.view(-1, 3, 36, 36)   # (B, 3, 36, 36)
        # x = torch.max(x, dim=1, keepdim=True)[0]    #(B, 1, 36, 36)

        x0 = x.unsqueeze(1)
        x1 = x0.clone()
        x2 = x0.clone()
        # x1 = x1.permute(0, 1, 3, 4, 2)
        # x2 = x2.permute(0, 1, 4, 2, 3)

        x0 = self.nn1(x0)
        x1 = self.nn2(x1)
        x2 = self.nn3(x2)

        x = torch.hstack([x0, x1, x2]) # B, 3, 36, 36
        x = torch.max(x, dim=1, keepdim=True)[0]

        x = F.relu(self.conv3(x))   # (B, 8, 36, 36)
        x = F.max_pool2d(x, 4)  # (B, 8, 9, 9)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VoxelCNNAns(nn.Module):
    "3D CNN for anistrophic (36x36x8) voxels"
    def __init__(self, num_classes):
        super(VoxelCNNAns, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, kernel_size=(6, 6, 8)) # Cx6x6x1
        x = x.view(-1, 576)

        x = F.relu(self.fc1(x))
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
    