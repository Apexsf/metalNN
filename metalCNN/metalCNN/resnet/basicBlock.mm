//
//  basicBlock.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/30.
//

#include "basicBlock.h"

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
    
    if (infoFromJson[@"conv3"]){
        NSDictionary* conv3Info = infoFromJson[@"conv3"];
        convParams conv3Param = makingConvParams(conv3Info[@"params"]);
        std::map<std::string, tensor> conv3Weights = makingConvWeight(conv3Info[@"weights"], conv3Param);
        
        NSDictionary* bn3Info = infoFromJson[@"bn3"];
        uint bn3Channel = makingBnParams(bn3Info[@"params"]);
        std::map<std::string, tensor> bn3Weights = makingBnWeight(bn3Info[@"weights"], bn3Channel);
        
        block.setDownSampleModule(conv3Param, bn3Channel, conv3Weights, bn3Weights);
    }
    
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


basicBlock::basicBlock(std::shared_ptr<gpuResource> resource, convParams& convPara1,
                       convParams& convPara2, uint bnChannel1, uint bnChannel2):resource_(resource),
conv1_(resource, "conv", convPara1),
conv2_(resource, "conv", convPara2), relu_(resource, "relu"), bn1_(resource, "bn", bnChannel1), bn2_(resource, "bn", bnChannel2), add_(resource, "elemWiseAdd"){
    
}

void basicBlock:: cleanDownSampleModule(){
    if (conv3P_) {
        delete conv3P_;
    }
    if (bn3P_) {
        delete bn3P_;
    }
    if(convConst3_){
        delete convConst3_;
    }
}

basicBlock::~basicBlock(){
    cleanDownSampleModule();
}

//basicBlock::basicBlock(std::shared_ptr<gpuResource> resource, NSDictionary* blockInfo){
//
//}


void basicBlock::loadWeights(std::map<std::string, tensor>& convWeight1, std::map<std::string, tensor>& convWeight2, std::map<std::string, tensor>& bnWeights1, std::map<std::string, tensor>& bnWeights2) {
    conv1_.loadWeight(convWeight1);
    conv2_.loadWeight(convWeight2);
    bn1_.loadWeight(bnWeights1);
    bn2_.loadWeight(bnWeights2);
}



void basicBlock::makingConstantAndShape(const shape& inShape) {
    convConst1_ = conv::makeConvConstant(inShape, conv1_.getParams());
    outShape1_ = shape{(uint)convConst1_.out_batch, conv1_.getParams().outC, (uint)convConst1_.out_height, (uint)convConst1_.out_width};
    bnConst1_ = bnConstant{convConst1_.out_batch, convConst1_.out_slice, convConst1_.out_height, convConst1_.out_width, convConst1_.out_size};
    
    actConst1_  = actConstant{convConst1_.out_batch, convConst1_.out_slice,
        convConst1_.out_height, convConst1_.out_width
    };
    
    convConst2_ = conv::makeConvConstant(outShape1_, conv2_.getParams());
    bnConst2_ = bnConstant {convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width, convConst2_.out_size};
    
    outShape1_ = shape{(uint)convConst1_.out_batch, (uint)conv1_.getParams().outC, (uint)convConst1_.out_height, (uint)convConst1_.out_width};
    outShape2_ = shape {(uint)convConst2_.out_batch, (uint)conv2_.getParams().outC, (uint)convConst2_.out_height, (uint)convConst2_.out_width};
    
    addConst_ = elemWiseConstant {convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width, convConst2_.out_size};
    
    actConst2_ = actConstant{convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width};
    
    if (hasDownSample_) {
        convConst3_ = new convConstant( conv::makeConvConstant(inShape, conv3P_->getParams()) );
        bnConst3_ = new bnConstant{convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width, convConst2_.out_size};
    }
}


id<MTLCommandBuffer> basicBlock::forward(const id<MTLBuffer> input,
                                         const shape& inShape,
                                         id <MTLBuffer> output,
                                         id<MTLCommandBuffer>* commandBufferP) {
    
    id<MTLCommandBuffer> commandBuffer = commandBufferP? *commandBufferP: [resource_->getCommandQueue() commandBuffer];
    
    makingConstantAndShape(inShape);
    
    
    // encode conv1
    id<MTLBuffer> interBuffer1 = resource_->getBuffer(convConst1_.out_batch * convConst1_.out_slice * 4 * convConst1_.out_size);
    std::vector<id<MTLBuffer>> inOutBuffers {input, interBuffer1};
    conv1_.encodeCommand(inOutBuffers, &convConst1_, commandBuffer);
    
    
    // encode bn1
    inOutBuffers = {interBuffer1, interBuffer1};
    bn1_.encodeCommand(inOutBuffers, &bnConst1_, commandBuffer);

 
    
    //encode relu
    inOutBuffers = {interBuffer1, interBuffer1};
    relu_.encodeCommand(inOutBuffers, &actConst1_, commandBuffer);

    
    // encode conv2
    id<MTLBuffer> interBuffer2 = resource_->getBuffer(convConst2_.out_batch * convConst2_.out_slice * 4 * convConst2_.out_size);
    inOutBuffers = {interBuffer1, interBuffer2};
    conv2_.encodeCommand(inOutBuffers, &convConst2_, commandBuffer);

    
    // encode bn2
    inOutBuffers = {interBuffer2, interBuffer2};
    bn2_.encodeCommand(inOutBuffers, &bnConst2_, commandBuffer);
    
    if(hasDownSample_){
        id<MTLBuffer> interBuffer3 = resource_->getBuffer(outShape2_.sizeNC4HW4());
        inOutBuffers = {input, interBuffer3};
        conv3P_->encodeCommand(inOutBuffers, (void*)convConst3_, commandBuffer);
        inOutBuffers = {interBuffer3, interBuffer3};
        bn3P_->encodeCommand(inOutBuffers, (void*) bnConst3_, commandBuffer);
        inOutBuffers = {interBuffer3, interBuffer2, output};
        add_.encodeCommand(inOutBuffers, &addConst_, commandBuffer);
//        resource_->putBuffer(interBuffer3);
    } else {
        // encode add
        inOutBuffers = {input, interBuffer2, output};
        add_.encodeCommand(inOutBuffers, &addConst_, commandBuffer);
    }

//    // encode relu
    inOutBuffers = {output, output};
    relu_.encodeCommand(inOutBuffers, &actConst2_, commandBuffer);
    
//    resource_->putBuffer(interBuffer1);
//    resource_->putBuffer(interBuffer2);
    
    
    return commandBuffer;
    
    
}

void basicBlock::setDownSampleModule(convParams& convPara3, uint bnChannel3,
                         std::map<std::string, tensor>& convWeight3,
                                     std::map<std::string, tensor>& bnWeight3) {
    cleanDownSampleModule();
    hasDownSample_ = true;
    conv3P_ = new conv(resource_, "conv", convPara3);
    conv3P_->loadWeight(convWeight3);
    bn3P_ = new bn(resource_, "bn", bnChannel3);
    bn3P_->loadWeight(bnWeight3);
}
