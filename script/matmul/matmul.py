import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)

r18 = resnet18(pretrained=True)
weight_dir = "./weights"
fc = r18.fc

weight = fc.weight.data
bias = fc.bias.data

weight.flatten().numpy().tofile("weight.bin")
bias.flatten().numpy().tofile("bias.bin")

x = torch.randn((512))
out = fc(x)

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')

print()