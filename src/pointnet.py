from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# Reference: https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 9)


    def forward(self, x):
        batchsize = x.size()[0]

        x = x.transpose(2, 1) # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0] # Global max pooling

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    

class STNkd(nn.Module):
    def __init__(self, k=32):
        super(STNkd, self).__init__()
        
        self.conv1 = nn.Conv1d(k, 32, 1)
        self.conv2 = nn.Conv1d(32, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, k*k)
        self.k = k


    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0] # Global max pooling

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    

class PointNetMini(nn.Module):
    def __init__(self, num_classes=10, feature_transform = False):
        super(PointNetMini, self).__init__()
        self.stn = STN3d()

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd()

        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 512, 1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # n_pts = x.size()[1] # (batch_size, num_points, 3)
        trans = self.stn(x)
        x = torch.bmm(x, trans)

        x = x.transpose(2, 1) # (batch_size, 3, num_points)
        x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)

        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0] # Global max pooling

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.feature_transform:
            return F.log_softmax(x, dim=1), trans_feat
        
        return F.log_softmax(x, dim=1)
        

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

