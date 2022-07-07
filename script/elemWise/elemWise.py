import torch
import numpy as np
import torch.nn as nn
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

x1 = torch.randn(2,11,71,83)
x2 = torch.randn(2,11,71,83)


out = x1 * x2

x1.detach().flatten().numpy().tofile("input1.bin")
x2.detach().flatten().numpy().tofile("input2.bin")
out.detach().flatten().numpy().tofile('out.bin')

print()

