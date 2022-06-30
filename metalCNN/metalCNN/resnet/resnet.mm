//
//  resnet.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/27.
//

#include "resnet.h"


basicBlock makingBasicBlock(std::shared_ptr<gpuResource> resource, NSDictionary *infoFromJson) {
    NSDictionary* conv1Info = infoFromJson[@"conv1"];
    convParams conv1Param = makingConvParams(conv1Info[@"params"]);
    std::map<std::string, tensor> conv1Weights = makingConvWeight(conv1Info[@"weights"], conv1Param);

    NSDictionary* conv2Info = infoFromJson[@"conv2"];
    convParams conv2Param = makingConvParams(conv2Info[@"params"]);
    std::map<std::string, tensor> conv2Weights = makingConvWeight(conv2Info[@"weights"], conv2Param);

    NSDictionary* bn1Info = infoFromJson[@"bn1"];
    uint bn1Channel = makingBnParams(bn1Info[@"params"]);
    std::map<std::string, tensor> bn1Weights = makingBnWeight(bn1Info[@"weights"], bn1Channel);
    
    NSDictionary* bn2Info = infoFromJson[@"bn2"];
    uint bn2Channel = makingBnParams(bn2Info[@"params"]);
    std::map<std::string, tensor> bn2Weights = makingBnWeight(bn2Info[@"weights"], bn2Channel);
    
    basicBlock block(resource, conv1Param, conv2Param, bn1Channel, bn2Channel);
    block.loadWeights(conv1Weights, conv2Weights, bn1Weights, bn2Weights);
    return block;
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
