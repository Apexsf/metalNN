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

data = {}

if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

def makeConvDict(conv, baseName):
    is_bias = conv.bias
    params = {"kernelH" : conv.kernel_size[0], "kernelW" : conv.kernel_size[1],
    "inC": conv.in_channels, "outC":conv.out_channels,
    "padH" : conv.padding[0], "padW" : conv.padding[1],
    "strideH" : conv.stride[0], "strideW" : conv.stride[1], "hasBias": True if is_bias
    else False}
    if is_bias: 
        bias = conv.bias.data.detach().flatten().float().tolist()

    weight_path = os.path.join(weight_dir, "{}_weignt.bin".format(baseName))
    weight_path = os.path.abspath(weight_path)
    conv.weight.data.detach().flatten().numpy().tofile(weight_path)
    bias_path = ""
    if is_bias:
        bias_path = os.path.join(weight_dir, "{}_bias.bin".format(baseName))
        bias_path = os.path.abspath(bias_path)
        conv.bias.data.detach().flatten().numpy().tofile(bias_path)


    weights = {"weight": weight_path,
    "bias" : bias_path}
    return {"weights" : weights, "params" : params}

def makeBnDict(bn, baseName):
    gamma = bn.weight.data
    beta = bn.bias.data
    running_mean =  bn.running_mean
    running_var = bn.running_var

    gamma_path = os.path.join(weight_dir, "{}_gamma.bin".format(baseName))
    beta_path = os.path.join(weight_dir, "{}_beta.bin".format(baseName))
    rm_path = os.path.join(weight_dir, "{}_rm.bin".format(baseName))
    rv_path = os.path.join(weight_dir, "{}_rv.bin".format(baseName))

    gamma_path = os.path.abspath(gamma_path)
    beta_path = os.path.abspath(beta_path)
    rm_path = os.path.abspath(rm_path)
    rv_path = os.path.abspath(rv_path)


    gamma.numpy().tofile(gamma_path)
    beta.numpy().tofile(beta_path)
    running_mean.numpy().tofile(rm_path)
    running_var.numpy().tofile(rv_path)


    params = {"channel": bn.num_features}
    weights = {"gamma" : gamma_path, "beta" : beta_path, "running_mean": rm_path, 
    "running_var": rv_path}
    return {"weights": weights, "params" : params}


def makePoolingDict(pooling, baseName):
    params = {"kernelH" : pooling.kernel_size, "kernelW" : pooling.kernel_size,
    "padH" : pooling.padding, "padW" : pooling.padding,
    "strideH": pooling.stride, "strideW": pooling.stride}
    return {"params": params}

# basic_block1 
conv1= r18.conv1
bn1 = r18.bn1
bn1.eval()
relu = r18.relu
maxpool = r18.maxpool

conv1Dict = makeConvDict(conv1,"conv")
bn1Dict = makeBnDict(bn1, "bn")
poolingDict = makePoolingDict(maxpool, "pooling")

preLayerDict = {
    "conv": conv1Dict,
    "bn" : bn1Dict,
    "pooling": poolingDict
}

with open("testData.json", 'w') as f:
    json.dump(preLayerDict, f)

x = torch.randn((2,3,100,100))
out = maxpool(relu(bn1(conv1(x))))
# out = bn1(conv1(x))
# out = conv1(x)
# out = relu(bn1(conv1(x)))
# out = conv1(x)
x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')

print()