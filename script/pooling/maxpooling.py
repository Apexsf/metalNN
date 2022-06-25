import torch
import numpy as np
import torch.nn as nn
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

x = torch.randn(2,11,71,83)
# x = torch.randn(2,7,16,16)
maxpooling = nn.MaxPool2d( kernel_size= (9,7), padding=(4,3), stride= (4,7))
# maxpooling = nn.MaxPool2d( kernel_size= (9,7), padding=(2,3), stride= (2,2))

out = maxpooling(x)

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')

print()

