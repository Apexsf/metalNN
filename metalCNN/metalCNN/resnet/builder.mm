//
//  builder.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/2.
//

#include "builder.h"

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



preLayer makingPreLayer(std::shared_ptr<gpuResource> resource, NSDictionary* infoFromJson) {
    return preLayer(
        resource,
        makingConv(resource, infoFromJson[@"conv"]),
        makingBN(resource, infoFromJson[@"bn"]),
        act(resource, "relu"),
        makingPooling(resource, "poolingMax", infoFromJson[@"pooling"])
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

