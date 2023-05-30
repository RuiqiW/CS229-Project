import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dataset import ImageDataset, PointCloudDataset, VoxelDataset, MultiViewDataset
from model import MLP, CNN, VanillaPointNet, VoxelCNN, MultiViewCNN
from pointnet import PointNetMini, feature_transform_regularizer

torch.manual_seed(1234)


DATA_FORMATS = ['single_view', 'multi_view', 'pcd', 'voxel']

PREFIX = 'train'

DATA_FORMAT = 'multi_view'
NUM_VIEWS = 4
ROOT_DIR = '../data/train_{}'.format(DATA_FORMAT)
DATA_LABELS = '../data/train_meshMNIST/labels.txt'


BATCH_SIZE = 128
EPOCHS = 100

USE_FEATURE_TRANSFORM = False

transform = transforms.Compose([
    transforms.ToTensor()
])


if __name__ == '__main__':
    
    writer = SummaryWriter()

    with open(DATA_LABELS, 'r') as f:
        lines = f.readlines()

    # # Shuffle the lines randomly
    # random.shuffle(lines)

    # Split the data into training, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    num_samples = len(lines)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    train_lines = lines[:num_train]
    val_lines = lines[num_train:num_train+num_val]
    test_lines = lines[num_train+num_val:]


    if DATA_FORMAT == 'single_view':
        filename_format = "{:05d}.png"
        train_dataset = ImageDataset(ROOT_DIR, train_lines, filename_format=filename_format, transform=transform)
        val_dataset = ImageDataset(ROOT_DIR, val_lines, filename_format=filename_format, transform=transform)
        test_dataset = ImageDataset(ROOT_DIR, test_lines, filename_format=filename_format, transform=transform)

        model = CNN(num_classes=10)

    elif DATA_FORMAT == 'multi_view':
        filename_format = "{:05d}.npy"
        train_dataset = MultiViewDataset(ROOT_DIR, train_lines, filename_format=filename_format, num_views=NUM_VIEWS)
        val_dataset = MultiViewDataset(ROOT_DIR, val_lines, filename_format=filename_format, num_views=NUM_VIEWS)
        test_dataset = MultiViewDataset(ROOT_DIR, test_lines, filename_format=filename_format, num_views=NUM_VIEWS)

        model = MultiViewCNN(num_classes=10, num_views=NUM_VIEWS)
    
    elif DATA_FORMAT == 'pcd':
        filename_format = "{:05d}.ply"
        train_dataset = PointCloudDataset(ROOT_DIR, train_lines, filename_format=filename_format, use_augmentation=True)
        val_dataset = PointCloudDataset(ROOT_DIR, val_lines, filename_format=filename_format)
        test_dataset = PointCloudDataset(ROOT_DIR, test_lines, filename_format=filename_format)

        # model = VanillaPointNet(num_classes=10)
        model = PointNetMini(num_classes=10, feature_transform=USE_FEATURE_TRANSFORM)

    elif DATA_FORMAT == 'voxel':
        filename_format = "{:05d}.npy"
        train_dataset = VoxelDataset(ROOT_DIR, train_lines, filename_format=filename_format)
        val_dataset = VoxelDataset(ROOT_DIR, val_lines, filename_format=filename_format)
        test_dataset = VoxelDataset(ROOT_DIR, test_lines, filename_format=filename_format)

        model = VoxelCNN(num_classes=10)
    else:
        raise NotImplementedError('Data format not supported')

    device = "cpu"
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    print("data format: ", DATA_FORMAT)
    print("model: ", model.__class__.__name__)


    for epoch in range(EPOCHS):
        print("Epoch {} / {}".format(epoch+1, EPOCHS))
        model.train()
        train_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            target =  target.to(device)
            optimizer.zero_grad()

            if USE_FEATURE_TRANSFORM:
                output, trans = model(data)
            else:
                output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if USE_FEATURE_TRANSFORM:
                loss += 0.001 * feature_transform_regularizer(trans)
            loss.backward()
            optimizer.step()

        print("Training Loss: {}, Training Accuracy: {}%".format( train_loss  / len(train_loader.dataset), 
              correct * 100 / len(train_loader.dataset)))
        writer.add_scalar('Training Loss', train_loss  / len(train_loader.dataset), epoch + 1)
        writer.add_scalar('Training Accuracy', correct / len(train_loader.dataset), epoch + 1)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
                data = data.to(device)
                target = target.to(device)
                if USE_FEATURE_TRANSFORM:
                    output, trans = model(data)
                else:
                    output = model(data)
                val_loss += criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        print("Validation Loss: {}, Validation Accuracy: {}%".format( val_loss  / len(val_loader.dataset), 
              correct * 100 / len(val_loader.dataset)))
        writer.add_scalar('Validation Loss', val_loss  / len(val_loader.dataset), epoch + 1)
        writer.add_scalar('Validation Accuracy', correct / len(val_loader.dataset), epoch + 1)
            


        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                data = data.to(device)
                target = target.to(device)
                if USE_FEATURE_TRANSFORM:
                    output, trans = model(data)
                else:
                    output = model(data)
                test_loss += criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, 
                                                                                len(test_loader.dataset), accuracy))