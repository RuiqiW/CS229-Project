import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

from dataset import PointCloudDataset
from model import VanillaPointNet


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
    
    root_dir = '../data/12_pcd/'
    image_list_file = '../data/12_meshMNIST/labels.txt'
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
    model = VanillaPointNet(num_classes=2)
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
    

