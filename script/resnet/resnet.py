import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42)


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

def makeFcDict(fc, baseName):
    weight = fc.weight.data
    bias = fc.bias.data
    weight_path =  os.path.join(weight_dir, "{}_fc.bin".format(baseName))
    bias_path = os.path.join(weight_dir, "{}_bias.bin".format(baseName))

    weight_path = os.path.abspath(weight_path)
    bias_path  = os.path.abspath(bias_path)
    weight.flatten().numpy().tofile(weight_path)
    bias.flatten().numpy().tofile(bias_path)

    params = {"inC" : fc.in_features,
    "outC": fc.out_features}
    weights = {"weight":weight_path, "bias": bias_path}
    return {"params": params, "weights":weights}
    

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


def makePreLayer(conv, bn, pooling, baseName):
    return {"conv" : makeConvDict(conv, "conv"), "bn" : makeBnDict(bn,"bn"),
    "pooling" : makePoolingDict(pooling, "pooling")}

def makePostLayer(fc, baseName):
    return {"fc" : makeFcDict(fc, baseName)}

def makeBasicBlock(basicBlock, baseName):
    basicBlockDict = {}
    conv1 = basicBlock.conv1
    conv2 = basicBlock.conv2
    bn1 = basicBlock.bn1
    bn2 = basicBlock.bn2
    basicBlockDict["conv1"] = makeConvDict(conv1, baseName + "_conv1")
    basicBlockDict["conv2"] = makeConvDict(conv2, baseName + "_conv2")
    basicBlockDict["bn1"] = makeBnDict(bn1, baseName + "_bn1")
    basicBlockDict["bn2"] = makeBnDict(bn2, baseName + "_bn2")
    if basicBlock.downsample is not None:
        basicBlockDict["conv3"] = makeConvDict(basicBlock.downsample[0], baseName + "_conv3")
        basicBlockDict["bn3"] = makeBnDict(basicBlock.downsample[1], baseName + "_bn3")
    return basicBlockDict

def makeBasicLayer(layer, baseName):
    layerDict = {}
    layerDict["basicBlock1"] = makeBasicBlock(layer[0], baseName + "_block1")
    layerDict["basicBlock2"] = makeBasicBlock(layer[1], baseName + "_block2")
    return layerDict

def makeResNet(resnet, baseName = "resnet"):
    preLayerDict = makePreLayer(resnet.conv1, resnet.bn1, resnet.maxpool,
    baseName + "_preLayer")
    layer1Dict = makeBasicLayer(resnet.layer1, baseName + "_layer1")
    layer2Dict = makeBasicLayer(resnet.layer2, baseName + "_layer2")
    layer3Dict = makeBasicLayer(resnet.layer3, baseName + "_layer3")
    layer4Dict = makeBasicLayer(resnet.layer4, baseName + "_layer4")
    postLayerDict = makePostLayer(resnet.fc, baseName + "_postLayer")
    return {
        "preLayer": preLayerDict,
        "basicLayer1": layer1Dict,
        "basicLayer2": layer2Dict,
        "basicLayer3": layer3Dict,
        "basicLayer4": layer4Dict,
        "postLayer": postLayerDict
    }
    print()


# basic_block1 
r18 = resnet18(pretrained=True)
r18 = r18.eval()

resnetDict = makeResNet(r18)

with open("testData.json", 'w') as f:
    json.dump(resnetDict, f)

x = torch.randn((1,3,256,256))

# out = r18.layer1(r18.maxpool(r18.relu(r18.bn1(r18.conv1(x)))))
# out =r18.layer2( r18.layer1(r18.maxpool(r18.relu(r18.bn1(r18.conv1(x))))))
# out =r18.layer3( r18.layer2( r18.layer1(r18.maxpool(r18.relu(r18.bn1(r18.conv1(x)))))))
out =r18.layer4(r18.layer3(r18.layer2(r18.layer1(r18.maxpool(r18.relu(r18.bn1(r18.conv1(x))))))))
out = r18.fc(r18.avgpool(out).squeeze())

x.detach().flatten().numpy().tofile("input.bin")
out.detach().flatten().numpy().tofile('out.bin')
print(out.shape)
print()
