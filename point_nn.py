import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

import os


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


class PointNet(torch.nn.Module):
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


def train(model, train_loader, optimizer, criterion, num_epoches, device):
    model.train()

    for epoch in range(num_epoches):
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / len(train_loader.dataset)

        print('Train Epoch: {} \t Loss: {:.6f} \t Accuracy: {}/{} ({:.0f}%)'.format(epoch+1, loss.item(), correct, 
                                                                                    len(train_loader.dataset), train_acc))


if __name__ == '__main__':
    
    root_dir = 'data/12_pcd/'
    image_list_file = 'data/12_meshMNIST/labels.txt'
    filename_format = "{:04d}.ply"

    # Set the split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Read the image IDs and labels from the text file
    with open(image_list_file, 'r') as f:
        lines = f.readlines()

    # # Shuffle the lines randomly
    # random.shuffle(lines)

    # Split the data into training, validation, and test sets
    num_samples = len(lines)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    train_lines = lines[:num_train]
    val_lines = lines[num_train:num_train+num_val]
    test_lines = lines[num_train+num_val:]

    # Create the datasets and data loaders
    train_dataset = PointCloudDataset(root_dir, train_lines, filename_format=filename_format)
    val_dataset = PointCloudDataset(root_dir, val_lines, filename_format=filename_format)
    test_dataset = PointCloudDataset(root_dir, test_lines, filename_format=filename_format)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Initialize the model and optimizer
    model = PointNet(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model for 10 epochs
    train(model, train_loader, optimizer, F.nll_loss, 10, device)

    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, 
                                                                             len(test_loader.dataset), accuracy))
    

