import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

r18 = resnet18(pretrained=True)

bn = nn.BatchNorm2d(16)
bn.eval()

x = torch.randn((2,16,64,64))
out = bn(x)

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')

gamma = bn.weight.data
beta = bn.bias.data
running_mean =  bn.running_mean
running_val = bn.running_var
print()