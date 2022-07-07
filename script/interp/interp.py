import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import os
import torch.nn.functional as F
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)
x = torch.randn((3,29,37,67))
x.detach().flatten().numpy().tofile("input.bin")
out = F.interpolate(x, size = (159,229),mode="bilinear")
out.detach().flatten().numpy().tofile('out.bin')
