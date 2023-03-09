import torch, math, torch.nn as nn, torch.nn.functional as F


"""Wide resnet code Based on code from https://github.com/yaodongyu/TRADES"""
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes) and stride == 1
        self.stride = stride
        self.convShortcut = (not self.equalInOut or stride > 1) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        # condition =  or self.stride > 1
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            stride_i = i == 0 and stride or 1 #if in_planes != out_planes else 1
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, stride_i, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ClassifierWRN(nn.Module):
    def __init__(self, depth=34, num_base=3, num_features = 2, widen_factor=10, dropRate=0.0, layer_num = 3, fac = 16, fft = 0):
        super(ClassifierWRN, self).__init__()
        print(dropRate)
        features = layer_num != -1
        self.fft = fft
        inChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        outChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.layer_num = layer_num
        # inChannels[layer_num] = num_base*inChannels[layer_num]
        assert(10%widen_factor == 0)
        if layer_num == 0:
            inChannels[layer_num] = num_features*inChannels[layer_num]
        elif layer_num in [1,2,3]:
            inChannels[layer_num] = num_features*inChannels[layer_num]*(10//widen_factor) ##Because the base mmodels were trained with wf = 10
        elif layer_num == 4:
            inChannels[-1] = 2*inChannels[-1]*(10//widen_factor)
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3*(features + 1), inChannels[0], kernel_size=3, stride=1, padding=1, bias=False) if self.layer_num == -1 else None
        # 1st block
        self.block1 = NetworkBlock(n, inChannels[0], outChannels[1], block, 1, dropRate) if self.layer_num <= 0 else None
        # 2nd block
        self.block2 = NetworkBlock(n, inChannels[1], outChannels[2], block, 2, dropRate) if self.layer_num <= 1 else None
        # 3rd block
        self.block3 = NetworkBlock(n, inChannels[2], outChannels[3], block, 2, dropRate) if self.layer_num <= 2 else None
        self.size_match_conv = nn.Conv2d(outChannels[3],128,5,2) if self.layer_num <= 2 else None
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(outChannels[3]) if self.layer_num <= 2 else None
        self.relu = nn.ReLU(inplace=True) if self.layer_num <= 2 else None
        self.fc = nn.Linear(inChannels[3], num_base) if self.layer_num <= 4 else None
        self.fc3 = nn.Linear(inChannels[3]*8*8, num_base) if self.layer_num == 3 else None
        self.nChannels = inChannels[3]
        self.outputs = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, out):
        self.outputs = []
        
        batch_size = out.shape[0]
        # [100, 3, 32, 32]
        out = self.conv1(out) if self.layer_num == -1 else out
        # [100, 16, 32, 32]
        out = self.block1(out) if self.layer_num <= 0 else out
        if self.layer_num == 0:
            self.outputs.append(out.cpu())
        # [100, 32, 32, 32]
        out = self.block2(out) if self.layer_num <= 1 else out
        if self.layer_num <=1:
            self.outputs.append(out.cpu())
        # [100, 64, 16, 16]
        out = self.block3(out) if self.layer_num <= 2 else out
        if self.layer_num <=2:
            self.outputs.append(out.cpu())
        # [100, 128, 8, 8]
        out = self.relu(self.bn1(out)) if self.layer_num <= 2 else out
        # [100, 128, 8, 8]
        if self.layer_num==3:
            out = out.view(batch_size, self.nChannels*8*8) if self.layer_num <= 3 else out
            out = self.fc3(out)
            self.outputs.append(out.cpu())
            return out
        else:
            out = F.avg_pool2d(out, 8) if self.layer_num <= 3 else out
            # [100, 128, 1, 1]
            out = out.view(batch_size, self.nChannels) if self.layer_num <= 3 else out
            # [100, 128]
        if self.layer_num <=3:
            self.outputs.append(out.cpu())
        out = self.fc(out)
        if self.layer_num <=4:
            self.outputs.append(out.cpu())
        # [100, 3]
        return out

class WideResNet(nn.Module):
    #wrn-40-2 or wrn-28-10
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, layer_num = 3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.layer_num = layer_num

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def forward_features(self, x):
        features = x.clone()
        # [100, 3, 32, 32]
        out = self.conv1(x); features = out.clone() if self.layer_num > -1 else features
        # [100, 16, 32, 32]
        out = self.block1(out); features = out.clone() if self.layer_num > 0 else features
        # [100, 32, 32, 32]
        out = self.block2(out); features = out.clone() if self.layer_num > 1 else features
        # [100, 64, 16, 16]
        out = self.block3(out); features = out.clone() if self.layer_num > 2 else features
        # [100, 128, 8, 8]
        out = self.relu(self.bn1(out)); features = out.clone() if self.layer_num > 2 else features
        # [100, 128, 8, 8]
        out = F.avg_pool2d(out, 8); features = out.clone() if self.layer_num > 3 else features
        # [100, 128, 1, 1]
        out = out.view(-1, self.nChannels); features = out.clone() if self.layer_num > 3 else features
        # [100, 128]
        out = self.fc(out)
        # [100, 10]
        return out, features

    def features(self, x, return_all = False):
        f_list = [] if return_all else None
        
        out = x

        if self.layer_num > -1 or return_all:
            out = self.conv1(x)
            if return_all:  f_list.append(out) 
        if self.layer_num > 0 or return_all:
            out = self.block1(out)
            if return_all:  f_list.append(out) 
        if self.layer_num > 1 or return_all:
            out = self.block2(out)
            if return_all:  f_list.append(out) 
        if self.layer_num > 2 or return_all:
            out = self.block3(out)
            if return_all:  f_list.append(out) 
            out = self.relu(self.bn1(out)) if self.layer_num > 2 else out
        out = F.avg_pool2d(out, 8) if self.layer_num > 3 else out
        out = out.view(-1, self.nChannels) if self.layer_num > 3 else out
        # f_list.append(self.fc(out)) 
        # out = self.fc(out)
        if return_all:
            return f_list
        return out


