import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import os


class ImageDataset(Dataset):
    def __init__(self, root_dir, image_list, filename_format, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for line in image_list:
            image_id, label = line.strip().split(',')
            self.images.append(filename_format.format(int(image_id)))
            self.labels.append(int(label)-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_id = self.images[index]
        label = self.labels[index]
        image_path = os.path.join(self.root_dir, image_id)

        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label



class PointCloudDataset(Dataset):
    def __init__(self, root_dir, point_list, filename_format):
        super().__init__()
        self.root_dir = root_dir
        self.pointclouds = []
        self.labels = []

        for line in point_list:
            pcd_id, label = line.strip().split(',')
            self.pointclouds.append(filename_format.format(int(pcd_id)))
            self.labels.append(int(label)-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pcd_id = self.pointclouds[index]
        label = self.labels[index]
        pcd_path = os.path.join(self.root_dir, pcd_id)

        pcd = o3d.io.read_point_cloud(pcd_path)
        points = torch.from_numpy(np.asarray(pcd.points)).float()

        return points, label

