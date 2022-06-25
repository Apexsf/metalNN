import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

r18 = resnet18(pretrained=True)

bn = r18.layer1[0].bn1
bn.eval()

x = torch.randn((2,64,67,67))
out = bn(x)

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')

gamma = bn.weight.data
beta = bn.bias.data
running_mean =  bn.running_mean
running_val = bn.running_var

gamma.numpy().tofile('gamma.bin')
beta.numpy().tofile('beta.bin')
running_mean.numpy().tofile('running_mean.bin')
running_val.numpy().tofile('running_var.bin')
# idx = 0
# a = (x[0,0,0,idx]- running_mean[idx]) / running_val[idx]**0.5 * gamma[idx] + beta[idx]
print()