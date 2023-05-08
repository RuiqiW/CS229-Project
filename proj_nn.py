import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


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



class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)
    

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Linear(56*56, 2048)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc(x.view(-1, 56*56))
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


transform = transforms.Compose([
    transforms.ToTensor()
])

# Set the paths and filenames
root_dir = 'data/12_proj/'
image_list_file = 'data/12_meshMNIST/labels.txt'
filename_format = "{:04d}.png"

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
train_dataset = ImageDataset(root_dir, train_lines, filename_format=filename_format, transform=transform)
val_dataset = ImageDataset(root_dir, val_lines, filename_format=filename_format, transform=transform)
test_dataset = ImageDataset(root_dir, test_lines, filename_format=filename_format, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Initialize the neural network model and optimizer
model = MLP(num_classes=2)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Train the model
model.train()
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch+1, loss.item()))


# Test the model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += nn.functional.nll_loss(output, target, size_average=False).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), accuracy))

