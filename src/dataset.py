import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import os


class ImageDataset(Dataset):
    def __init__(self, root_dir, data_list, filename_format, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for line in data_list:
            data_id, label = line.strip().split(',')
            self.images.append(filename_format.format(int(data_id)))
            self.labels.append(int(label)-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data_id = self.images[index]
        label = self.labels[index]
        data_path = os.path.join(self.root_dir, data_id)

        with open(data_path, 'rb') as f:
            image = Image.open(f).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label



class PointCloudDataset(Dataset):
    def __init__(self, root_dir, data_list, filename_format):
        super().__init__()
        self.root_dir = root_dir
        self.pointclouds = []
        self.labels = []

        for line in data_list:
            data_id, label = line.strip().split(',')
            self.pointclouds.append(filename_format.format(int(data_id)))
            self.labels.append(int(label)-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_id = self.pointclouds[index]
        label = self.labels[index]
        data_path = os.path.join(self.root_dir, data_id)

        pcd = o3d.io.read_point_cloud(data_path)
        points = torch.from_numpy(np.asarray(pcd.points)).float()

        return points, label


class VoxelDataset(Dataset):
    def __init__(self, root_dir, data_list, filename_format):
        super().__init__()
        self.root_dir = root_dir
        self.voxels = []
        self.labels = []

        for line in data_list:
            data_id, label = line.strip().split(',')
            self.voxels.append(filename_format.format(int(data_id)))
            self.labels.append(int(label)-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_id = self.voxels[index]
        label = self.labels[index]
        data_path = os.path.join(self.root_dir, data_id)

        with open(data_path, 'rb') as f:
            voxels = np.load(f)

        voxels = torch.from_numpy(voxels).float()

        return voxels, label

