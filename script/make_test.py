import torch
import numpy as np
import torch.nn as nn

torch.manual_seed(42)

x = torch.randn(2,4,64,64)
conv = nn.Conv2d(in_channels=4, out_channels = 8, kernel_size= 3, padding=1, stride= 2,bias=False)
out = conv(x)

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')
conv.weight.data.detach().flatten().numpy().tofile('weight.bin')
print()