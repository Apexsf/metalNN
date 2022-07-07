import torch
import numpy as np
import torch.nn as nn
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

x = torch.randn(2,11,71,83)
conv = nn.Conv2d(in_channels=11, out_channels = 9, kernel_size= (5,3), padding=(4,9), stride= (4,7),bias=True)
out = conv(x)

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')
conv.weight.data.detach().flatten().numpy().tofile('weight.bin')
conv.bias.data.detach().flatten().numpy().tofile('bias.bin')
print()

