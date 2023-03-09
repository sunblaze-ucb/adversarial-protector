import torch
import torch.nn as nn
import torch.multiprocessing as _mp
import torch.nn.functional as F

'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ClassifierResnet(nn.Module):
    def __init__(self, block, num_blocks, layer_num = 4, num_attacks=3, fac=16):
        super(ClassifierResnet, self).__init__()
        #layer_num = 1,2,3,4 -> indicating immediately after layer1, layer2, ....
        self.in_planes = [64, 64, 128, 256, 512]
        self.out_planes = [64, 128, 256, 512, 1024]
        self.in_planes[layer_num] = self.in_planes[layer_num]*num_attacks
        self.out_planes[layer_num - 1] = self.out_planes[layer_num - 1]*3 if layer_num != 0 else self.out_planes[layer_num - 1]

        self.layer_num = layer_num
        self.fac = fac
        assert (self.layer_num >= 0)
        if layer_num<=3:
            assert (fac==1)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, self.in_planes[0], self.out_planes[0], num_blocks[0], stride=1) if self.layer_num == 0 else None
        self.layer2 = self._make_layer(block, self.in_planes[1], self.out_planes[1], num_blocks[1], stride=2) if self.layer_num <= 1 else None
        self.layer3 = self._make_layer(block, self.in_planes[2], self.out_planes[2], num_blocks[2], stride=2) if self.layer_num <= 2 else None
        self.layer4 = self._make_layer(block, self.in_planes[3], self.out_planes[3], num_blocks[3], stride=2) if self.layer_num <= 3 else None
        self.linear1 = nn.Linear(self.in_planes[4]*fac, self.out_planes[4])
        self.linear2 = nn.Linear(1024, num_attacks)

    def forward(self, out):
        # [100, 3, 32, 32]
        # out = self.conv1(x) 
        # [100, 64, 32, 32]
        out = self.layer1(out) if self.layer_num == 0 else out
        # [100, 64*3, 32, 32]
        out = self.layer2(out) if self.layer_num <= 1 else out
        # [100, 128*3, 16, 16]
        out = self.layer3(out) if self.layer_num <= 2 else out
        # [100, 256*3, 8, 8]
        out = self.layer4(out) if self.layer_num <= 3 else out
        # [100, 512*3, 4, 4]
        out = F.avg_pool2d(out, 4) if self.fac == 1 else out
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out 

    def _make_layer(self, block, inplanes, outplanes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(inplanes, outplanes, stride))
            inplanes = outplanes * block.expansion
        return nn.Sequential(*layers)

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, layer_num = 4):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.layer_num = layer_num

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_features(self,x):
        # [100, 3, 32, 32]
        out = self.conv1(x); features = out.clone()
        # [100, 64, 32, 32]
        out = self.layer1(out); features = out.clone() if self.layer_num > 0 else features
        # [100, 64, 32, 32]
        out = self.layer2(out); features = out.clone() if self.layer_num > 1 else features
        # [100, 128, 16, 16]
        out = self.layer3(out); features = out.clone() if self.layer_num > 2 else features
        # [100, 256, 8, 8]
        out = self.layer4(out); features = out.clone() if self.layer_num > 3 else features
        # [100, 512, 4, 4]
        out = F.avg_pool2d(out, 4)
        # [100, 512, 1, 1]
        out = out.view(out.size(0), -1)
        # [100, 512]
        out = self.linear(out)
        return out, features

    def features(self,x):
        # [100, 3, 32, 32]
        out = self.conv1(x); features = out.clone()
        # [100, 64, 32, 32]
        out = self.layer1(out); features = out.clone() if self.layer_num > 0 else features
        # [100, 64, 32, 32]
        out = self.layer2(out); features = out.clone() if self.layer_num > 1 else features
        # [100, 128, 16, 16]
        out = self.layer3(out); features = out.clone() if self.layer_num > 2 else features
        # [100, 256, 8, 8]
        out = self.layer4(out); features = out.clone() if self.layer_num > 3 else features

        return features

def PreActResNet18(num_classes = 10, layer_num = 4):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes, layer_num)
