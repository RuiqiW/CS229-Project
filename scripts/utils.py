import torch

import sys
sys.path.append("../src")
from model import MLP, CNN, VanillaPointNet, VoxelCNN, MultiViewCNN
from pointnet import PointNetMini

from torchsummary import summary

import sklearn.metrics as metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(params)
    return sum(params)


# def draw_confusion_matrix(pred, target):



if __name__ == '__main__':
    # # model = PointNetMini(num_classes=10, feature_transform=True)
    # model = MultiViewCNN(num_classes=10)
    # print(count_parameters(model))


    pred = torch.load('../src/pcd_pred.pt').numpy()
    target = torch.load('../src/target.pt').numpy()

    N = 10

    cf_matrix = metrics.confusion_matrix(target, pred)
    cf_sum = np.sum(cf_matrix)

    counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    percentages = ["{0:.2f}%".format(value) for value in cf_matrix.flatten() / cf_sum]

    labels = ["{}\n{}\n".format(v1, v2) for v1, v2 in zip(counts, percentages)]
    labels = np.asarray(labels).reshape(N, N)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion matrix for axis-aligned point clouds - PointNet wo feature trans')
    # plt.show()
    plt.savefig('../plots/cf_matrix_pcd.png')