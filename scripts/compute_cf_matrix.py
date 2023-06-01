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




if __name__ == '__main__':
    pred = torch.load('../src/pcd_axis_aligned_pred.pt').numpy()
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
    plt.title('Confusion matrix for axis-aligned point clouds - Point Net (wo feature trans)')
    # plt.show()
    plt.savefig('../plots/cf_matrix_axis_aligned_pcd.png')