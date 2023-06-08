import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch3d.transforms as transform3d

from PIL import Image

import os
import copy


class ImageDataset(Dataset):
    def __init__(self, root_dir, data_list, filename_format, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for line in data_list:
            data_id, label = line.strip().split(',')
            self.images.append(filename_format.format(int(data_id)))
            self.labels.append(int(label))

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


class MultiViewDataset(Dataset):
    def __init__(self, root_dir, data_list, filename_format, num_views=12):
        super().__init__()
        self.num_views = num_views
        self.root_dir = root_dir
        self.images = []
        self.labels = []

        for line in data_list:
            data_id, label = line.strip().split(',')
            self.images.append(filename_format.format(int(data_id)))
            self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_id = self.images[index]
        label = self.labels[index]
        data_path = os.path.join(self.root_dir, data_id)

        with open(data_path, 'rb') as f:
            images = np.load(f)

        images = images[:self.num_views, :, :]
        images = torch.from_numpy(images).float()

        return images, label
    

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, data_list, filename_format, use_augmentation=False, use_feature=False):
        super().__init__()
        self.root_dir = root_dir
        self.pointclouds = []
        self.labels = []
        self.use_augmentation = use_augmentation
        self.use_feature = use_feature

        for line in data_list:
            data_id, label = line.strip().split(',')
            self.pointclouds.append(filename_format.format(int(data_id)))
            self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_id = self.pointclouds[index]
        label = self.labels[index]
        data_path = os.path.join(self.root_dir, data_id)

        pcd = o3d.io.read_point_cloud(data_path)

        if not self.use_feature:
            points = torch.from_numpy(np.asarray(pcd.points)).float()
            points = points - torch.mean(points, dim=0)

            if self.use_augmentation:
                rotation = transform3d.random_rotation()
                T = transform3d.Rotate(rotation)
                points = T.transform_points(points)
        else:
            features = self.get_fpfh_feature(pcd)
            points = torch.from_numpy(copy.deepcopy(features.data)).float() # 33, P
            points = torch.transpose(points, 0, 1)

        return points, label
    

    def get_fpfh_feature(self, pcd, voxel_size=0.02, radius_normal_factor=2, radius_feature_factor=2):
        # pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down = pcd
        
        radius_normal = voxel_size * radius_normal_factor
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = voxel_size * radius_feature_factor
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        return pcd_fpfh


class VoxelDataset(Dataset):
    def __init__(self, root_dir, data_list, filename_format):
        super().__init__()
        self.root_dir = root_dir
        self.voxels = []
        self.labels = []

        for line in data_list:
            data_id, label = line.strip().split(',')
            self.voxels.append(filename_format.format(int(data_id)))
            self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_id = self.voxels[index]
        label = self.labels[index]
        data_path = os.path.join(self.root_dir, data_id)

        with open(data_path, 'rb') as f:
            voxels = np.load(f)
            if voxels.shape[0] == 44:
                voxels = voxels[4:40, 4:40, 4:40]

        voxels = torch.from_numpy(voxels).float()

        return voxels, label

