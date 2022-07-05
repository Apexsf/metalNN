//
//  builder.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/2.
//

#include "builder.h"

resnet makingResNet(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson){
    preLayer prel = makingPreLayer(resource, infoFromJson[@"preLayer"]);
    basicLayer bl1 = makingBasicLayer(resource, infoFromJson[@"basicLayer1"]);
    basicLayer bl2 = makingBasicLayer(resource, infoFromJson[@"basicLayer2"]);
    basicLayer bl3 = makingBasicLayer(resource, infoFromJson[@"basicLayer3"]);
    basicLayer bl4 = makingBasicLayer(resource, infoFromJson[@"basicLayer4"]);
    postLayer postl = makingPostLayer(resource, infoFromJson[@"postLayer"]);
    resnet net(resource, prel, bl1, bl2, bl3, bl4, postl);
    return net;
}

basicBlock makingBasicBlock(std::shared_ptr<gpuResource> resource, NSDictionary *infoFromJson) {
    conv conv1 = makingConv(resource, infoFromJson[@"conv1"]);
    conv conv2 = makingConv(resource, infoFromJson[@"conv2"]);
    bn bn1 = makingBN(resource, infoFromJson[@"bn1"]);
    bn bn2 = makingBN(resource, infoFromJson[@"bn2"]);
    basicBlock block (resource, conv1, conv2 , bn1, bn2);

    if (infoFromJson[@"conv3"]){
        conv conv3 = makingConv(resource, infoFromJson[@"conv3"]);
        bn bn3 = makingBN(resource, infoFromJson[@"bn3"]);
        block.setDownSampleModule(conv3, bn3);
    }
    
    return block;
}

basicLayer makingBasicLayer(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson) {
    return  {
        makingBasicBlock(resource, infoFromJson[@"basicBlock1"]),
        makingBasicBlock(resource, infoFromJson[@"basicBlock2"])
    };
}

postLayer makingPostLayer(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson){
    return postLayer(
                     resource,
                     pooling(resource, "poolingAvg", poolingParams{0,0,0,0,0,0}),
                     makingFC(resource, infoFromJson[@"fc"])
                     );
}


interp makingInterp(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson) {
    return interp(resource, "interpBilinear",
                  {interpParams::targetMode::SIZE,
     uint([infoFromJson[@"H"] intValue]),
        uint([infoFromJson[@"W"] intValue]),
        0,0});
}

preLayer makingPreLayer(std::shared_ptr<gpuResource> resource, NSDictionary* infoFromJson) {
    return preLayer(
        resource,
        makingConv(resource, infoFromJson[@"conv"]),
        makingBN(resource, infoFromJson[@"bn"]),
        act(resource, "relu"),
        makingPooling(resource, "poolingMax", infoFromJson[@"pooling"]),
        makingInterp(resource, infoFromJson[@"inputSize"])
    );
}



pooling makingPooling(std::shared_ptr<gpuResource>resource, std::string poolingMode, NSDictionary* infoFromJson){
    pooling p (resource, poolingMode, makingPoolingParams(infoFromJson[@"params"]));
    return p;
}

poolingParams makingPoolingParams(NSDictionary* poolingParamsInfo) {
    return  poolingParams{
        uint([poolingParamsInfo[@"kernelH"] intValue]),
        uint([poolingParamsInfo[@"kernelW"] intValue]),
        uint([poolingParamsInfo[@"padH"] intValue]),
        uint([poolingParamsInfo[@"padW"] intValue]),
        uint([poolingParamsInfo[@"strideH"] intValue]),
        uint([poolingParamsInfo[@"strideW"] intValue]),
    };
}

conv makingConv(std::shared_ptr<gpuResource> resource, NSDictionary *infoFromJson) {
    conv c(resource, "conv", makingConvParams(infoFromJson[@"params"]));
    auto weights = makingConvWeight(infoFromJson[@"weights"], c.getParams());
    c.loadWeight(weights);
    return c;
    
}

bn makingBN(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson) {
    bn b(resource, "bn", makingBnParams(infoFromJson[@"params"]));
    auto weights = makingBnWeight(infoFromJson[@"weights"], b.getParams());
    b.loadWeight(weights);
    return b;
}

matmul makingFC(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson){
    matmul m(resource, uint([infoFromJson[@"params"][@"inC"] intValue]),
              uint([infoFromJson[@"params"][@"outC"] intValue]));
    auto weights = makingFCWeight(infoFromJson[@"weights"], m.getInC(), m.getOutC());
    m.loadWeight(weights);
    return m;
    
}
convParams makingConvParams(NSDictionary* convParamsInfo) {
    convParams params{
        uint([convParamsInfo[@"kernelH"] intValue]),
        uint([convParamsInfo[@"kernelW"] intValue]),
        uint([convParamsInfo[@"inC"] intValue]),
        uint([convParamsInfo[@"outC"] intValue]),
        uint([convParamsInfo[@"padH"] intValue]),
        uint([convParamsInfo[@"padW"]  intValue]),
        uint([convParamsInfo[@"strideH"]  intValue]),
        uint([convParamsInfo[@"strideW"]  intValue]),
    
    };
    return params;
}

std::map<std::string, tensor> makingFCWeight (NSDictionary* fcWeightInfo,
                                               uint inC, uint outC){
    
    tensor weight(1,1, outC, inC);
    tensor bias(1,outC,1,1);
    weight.loadFromFile([fcWeightInfo[@"weight"] cString]);
    bias.loadFromFile([fcWeightInfo[@"bias"] cString]);
    std::map<std::string, tensor> weights = {
        {"weight", std::move(weight)},
        {"bias", std::move(bias)}
    };
    return weights;
}

std::map<std::string, tensor> makingConvWeight (NSDictionary* convWeightInfo, convParams params) {
 
    tensor weight(params.outC, params.inC, params.kernelH, params.kernelW);
    weight.loadFromFile( [convWeightInfo[@"weight"] cString]);

    if (strcmp([convWeightInfo[@"bias"] cString], "") != 0) {
        tensor bias (1,params.outC,1,1);
        bias.loadFromFile([convWeightInfo[@"bias"] cString]);
        std::map<std::string, tensor> weights = {
            {"weights", std::move(weight)},
            {"bias" , std::move(bias)}
        };
        return weights;
        
    } else {
        std::map<std::string, tensor> weights = {
            {"weight", std::move(weight)}
        };
        return weights;
    }

}


uint makingBnParams (NSDictionary* bnParamsInfo) {
    return uint([bnParamsInfo[@"channel"] intValue]);
}

std::map<std::string, tensor>  makingBnWeight (NSDictionary* bnWeightInfo,
                                               uint params) {
    tensor gamma(1, params, 1, 1);
    tensor beta(1, params, 1, 1);
    tensor rm(1,params,1,1);
    tensor rv(1, params,1,1);
    gamma.loadFromFile([bnWeightInfo[@"gamma"] cString]);
    beta.loadFromFile([bnWeightInfo[@"beta"] cString]);
    rm.loadFromFile([bnWeightInfo[@"running_mean"] cString]);
    rv.loadFromFile([bnWeightInfo[@"running_var"] cString]);
    
    std::map<std::string, tensor> weights = {
        {"gamma", std::move(gamma)},
        {"beta", std::move(beta)},
        {"running_mean", std::move(rm)},
        {"running_var", std::move(rv)}
    };
    return weights;
}

