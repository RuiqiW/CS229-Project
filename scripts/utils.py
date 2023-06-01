import torch

import sys
sys.path.append("../src")
from model import MLP, CNN, VanillaPointNet, VoxelCNN, MultiViewCNN, VoxelCNNProbing, VoxelCNNProbingMul
from pointnet import PointNetMini


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(params)
    return sum(params)



if __name__ == '__main__':
    # # model = PointNetMini(num_classes=10, feature_transform=True)
    model = VoxelCNNProbingMul(num_classes=10)
    print(count_parameters(model))
