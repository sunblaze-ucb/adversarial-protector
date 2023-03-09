import torch
import torch.nn as nn
import torch.multiprocessing as _mp
import torch.nn.functional as F


class CNN_Cifar(nn.Module):
    def __init__(self, layer_num = -1):
        super(CNN_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.layer_num = layer_num

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.maxpool(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)  
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_features(self,x):
        features = x.clone()
        x = nn.ReLU()(self.conv1(x))
        x = self.maxpool(x); features = x.clone() if self.layer_num == 1 else features
        x = nn.ReLU()(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1); features = x.clone() if self.layer_num == 2 else features
        x = nn.ReLU()(self.fc1(x)); features = x.clone() if self.layer_num == 3 else features
        x = nn.ReLU()(self.fc2(x)); features = x.clone() if self.layer_num == 4 else features
        x = self.fc3(x)
        return x, features

    def features(self,x):
        x = nn.ReLU()(self.conv1(x)) if self.layer_num > 0 else x
        x = self.maxpool(x) if self.layer_num > 0 else x
        x = nn.ReLU()(self.conv2(x)) if self.layer_num > 1 else x
        x = self.maxpool(x) if self.layer_num > 2 else x
        x = x.view(x.shape[0], -1) if self.layer_num > 2 else x
        x = nn.ReLU()(self.fc1(x)) if self.layer_num > 2 else x
        x = nn.ReLU()(self.fc2(x)) if self.layer_num > 3 else x
        x = self.fc3(x) if self.layer_num > 2 else x
        return x

class ClassifierCNN_Cifar(nn.Module):
    def __init__(self, num_attacks=3, layer_num = 1, fac = 1):
        super(ClassifierCNN, self).__init__()
        assert (layer_num in [0,1,2,3,4])
        # self.in_planes = [1, 32, 7*7*64, 1024]
        self.in_planes = [3, 32, 5*5*64, 120, 84]
        self.out_planes = [32, 64, 120, 84, num_attacks]

        self.in_planes[layer_num] = self.in_planes[layer_num]*3
        self.out_planes[layer_num - 1] = self.out_planes[layer_num - 1]*3 if layer_num != 0 else self.out_planes[layer_num - 1]

        assert (fac==1 or layer_num!=3)
        if layer_num != 3:
            self.in_planes[layer_num + 1] = self.in_planes[layer_num + 1]*fac
            self.out_planes[layer_num] = self.out_planes[layer_num]*fac

        self.layer_num = layer_num
        self.conv1 = nn.Conv2d(self.in_planes[0], self.out_planes[0], 5, padding = 2) if self.layer_num == 0 else None
        self.conv2 = nn.Conv2d(self.in_planes[1], self.out_planes[1], 5, padding = 2) if self.layer_num <= 1 else None
        self.fc1 = nn.Linear(self.in_planes[2], self.out_planes[2]) if self.layer_num <= 2 else None
        self.fc2 = nn.Linear(self.in_planes[3], self.out_planes[3]) if self.layer_num <= 3 else None
        self.fc3 = nn.Linear(self.in_planes[4], self.out_planes[4])
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self,x):
        x = nn.ReLU()(self.conv1(x)) if self.layer_num == 0 else x
        x = self.maxpool(x) if self.layer_num == 0 else x
        x = nn.ReLU()(self.conv2(x)) if self.layer_num <= 1 else x
        x = self.maxpool(x) if self.layer_num <= 1 else x
        x = x.view(x.shape[0], -1)  if self.layer_num <= 1 else x
        x = nn.ReLU()(self.fc1(x)) if self.layer_num <= 2 else x
        x = nn.ReLU()(self.fc2(x)) if self.layer_num <= 3 else x
        x = self.fc3(x) if self.layer_num <= 3 else x
        return x



