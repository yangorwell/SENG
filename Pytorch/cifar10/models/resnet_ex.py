# ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_ex import Conv2dEx, LinearEx, BatchNorm2dEx

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dEx(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2dEx(planes)
        self.conv2 = Conv2dEx(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2dEx(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dEx(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2dEx(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2dEx(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2dEx(planes, momentum=0.99)
        self.conv2 = Conv2dEx(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2dEx(planes, momentum=0.99)
        self.conv3 = Conv2dEx(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2dEx(self.expansion*planes, momentum=0.99)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dEx(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2dEx(self.expansion*planes, momentum=0.99)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2dEx(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2dEx(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = LinearEx(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2dEx):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2dEx):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if False:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)
