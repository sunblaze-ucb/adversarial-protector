import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ClassifierCNN(nn.Module):
    #Based on : https://github.com/yaodongyu/TRADES/blob/master/models/small_cnn.py
    def __init__(self, dropout=0.5, layer_num = -1, num_base = 2, num_features = 0):
        super(ClassifierCNN, self).__init__()

        self.layer_num = layer_num
        activ = nn.ReLU(True)
        self.num_features = num_features
        self.feature_extractor_1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),]))

        inp_channels = 32*self.num_features if self.layer_num == 0 else 32
        self.feature_extractor_2 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(inp_channels, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        classifier_input_shape = self.num_features * 64 * 4 * 4  if self.layer_num == 1 else 64 * 4 * 4
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(classifier_input_shape, 200)),
            ('relu1', activ),
            # ('drop', nn.Dropout(dropout)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, num_base)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.feature_extractor_1(x) if self.layer_num==-1 else x
        x = self.feature_extractor_2(x) if self.layer_num<=0 else x
        logits = self.classifier(x.view(batch_size,-1))
        return logits


class CNN(nn.Module):
    #https://github.com/yaodongyu/TRADES/blob/master/models/small_cnn.py
    def __init__(self, dropout=0.5, layer_num = -1):
        super(CNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10
        self.layer_num = layer_num

        activ = nn.ReLU(True)

        self.feature_extractor_1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),]))

        self.feature_extractor_2 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(dropout)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, inp):
        f1 = self.feature_extractor_1(inp)
        features = self.feature_extractor_2(f1)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

    def forward_features(self, inp):
        f1 = self.feature_extractor_1(inp)
        features = self.feature_extractor_2(f1)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        ret = (logits, f1) if self.layer_num == 0 else (logits,features)
        return ret

    def features(self, inp, return_all = False):
        f_list = [] if return_all else None
        f1 = self.feature_extractor_1(inp) 
        if return_all:  f_list.append(f1)
        features = self.feature_extractor_2(f1)
        if return_all: f_list.append(features)
        ret = f1 if self.layer_num == 0 else features
        if return_all:
            return f_list
        return ret
    


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

def CNN_MSD():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))
