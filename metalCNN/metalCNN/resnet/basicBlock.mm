//
//  basicBlock.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/30.
//

#include "basicBlock.h"

basicBlock::basicBlock(std::shared_ptr<gpuResource> resource, convParams& convPara1,
                       convParams& convPara2, uint bnChannel1, uint bnChannel2):resource_(resource),
conv1_(resource, "conv", convPara1),
conv2_(resource, "conv", convPara2), relu_(resource, "relu"), bn1_(resource, "bn", bnChannel1), bn2_(resource, "bn", bnChannel2), add_(resource, "elemWiseAdd"){
    
}


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

    
    // encode add
    inOutBuffers = {input, interBuffer2, output};
    add_.encodeCommand(inOutBuffers, &addConst_, commandBuffer);


    // encode relu
    inOutBuffers = {output, output};
    relu_.encodeCommand(inOutBuffers, &actConst2_, commandBuffer);
    
    resource_->putBuffer(interBuffer1);
    resource_->putBuffer(interBuffer2);
    
    return commandBuffer;
    
    
}
