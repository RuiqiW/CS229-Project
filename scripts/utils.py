import torch

import sys
sys.path.append("../src")
from model import MLP, CNN, VanillaPointNet, VoxelCNN, MultiViewCNN
from pointnet import PointNetMini

from torchsummary import summary

# model = PointNetMini(num_classes=10, feature_transform=True)
model = MultiViewCNN(num_classes=10)

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(params)
    return sum(params)

print(count_parameters(model))