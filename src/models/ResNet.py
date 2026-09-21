'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.dropout = 0.0

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, filter=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.filter = filter

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_embedding=False, dropout=0.0, reverse_grad=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = F.dropout(out, p=dropout, training=True)
        out = self.layer2(out)
        out = F.dropout(out, p=dropout, training=True)
        out = self.layer3(out)
        out = F.dropout(out, p=dropout, training=True)
        out = self.layer4(out)
        out = F.dropout(out, p=dropout, training=True)
        out = F.avg_pool2d(out, 4)
        embeddings = out.view(out.size(0), -1)
        if reverse_grad:
            embeddings = ReverseLayerF.apply(embeddings)
        out = self.fc(embeddings)
        
        if self.filter is not None: # for imagenetR
            out = out[:, self.filter]
            
        if get_embedding:
            return out, embeddings
        return out
    
        
                    
def ResNet18(filter=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], filter=filter)


def ResNetDropout18(filter=None):
    return ResNetDropout(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], filter=filter)

class ResNetDropout(torchvision.models.resnet.ResNet):
    """
    For pretrained ResNet models from Torchvision.
    """
    def __init__(self, block, num_blocks, filter=None):
        super(ResNetDropout, self).__init__(block, num_blocks)
        self.filter=filter
        
    def forward(self, x: torch.Tensor, dropout=0.0, get_embedding=False, reverse_grad=False) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.layer2(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.layer3(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.layer4(x)
        x = F.dropout(x, p=dropout, training=True)

        x = self.avgpool(x)
        embedding = torch.flatten(x, 1)
        
        if reverse_grad:
            embedding = ReverseLayerF.apply(embedding)
            
        x = self.fc(embedding)
        
        if self.filter is not None: # for imagenetR
            x = x[:, self.filter]
        
        if get_embedding:
            return x, embedding
        else:
            return x


class ReverseLayerF(Function):
	"""
	Gradient negation utility class
	"""				 
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None
