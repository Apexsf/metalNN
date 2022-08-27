import torch
import numpy as np
import torch.nn as nn
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

x = torch.randn(2,11,71,83)
conv = nn.Conv2d(in_channels=11, out_channels = 9, kernel_size= (5,3), padding=(4,9), stride= (4,7),bias=True)
bn = nn.BatchNorm2d(num_features= 9)
## set weights of bn to random
bn.weight.data = torch.rand(9) 
bn.bias.data = torch.rand(9)
running_mean = torch.rand(9)
running_val = torch.rand(9)
bn.eval()
out = bn(conv(x))
# out = conv(x)

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')
conv.weight.data.detach().flatten().numpy().tofile('weight.bin')
conv.bias.data.detach().flatten().numpy().tofile('bias.bin')

gamma = bn.weight.data
beta = bn.bias.data
running_mean =  bn.running_mean
running_val = bn.running_var

gamma.numpy().tofile('gamma.bin')
beta.numpy().tofile('beta.bin')
running_mean.numpy().tofile('running_mean.bin')
running_val.numpy().tofile('running_var.bin')
print(gamma)

