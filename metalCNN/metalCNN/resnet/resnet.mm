//
//  resnet.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/27.
//

#include "resnet.h"


basicBlock::basicBlock(std::shared_ptr<gpuResource> resource, convParams& convPara1,
                       convParams& convPara2, uint bnChannel1, uint bnChannel2):resource_(resource),
conv1_(resource, "conv", convPara1),
conv2_(resource, "conv", convPara2), relu_(resource, "relu"), bn1_(resource, "bn", bnChannel1), bn2_(resource, "bn", bnChannel2){
    
}


void basicBlock::loadWeights(std::map<std::string, tensor>& convWeight1, std::map<std::string, tensor>& convWeight2, std::map<std::string, tensor>& bnWeights1, std::map<std::string, tensor>& bnWeights2) {
    conv1_.loadWeight(convWeight1);
    conv2_.loadWeight(convWeight2);
    bn1_.loadWeight(bnWeights1);
    bn2_.loadWeight(bnWeights2);
}



void basicBlock::makingConstantAndShape(shape& inShape) {
    convConst1_ = conv::makeConvConstant(inShape, conv1_.getParams());
    outShape1_ = shape{(uint)convConst1_.out_batch, conv1_.getParams().outC, (uint)convConst1_.out_height, (uint)convConst1_.out_width};
    bnConst1_ = bnConstant{convConst1_.out_batch, convConst1_.out_slice, convConst1_.out_height, convConst1_.out_width, convConst1_.out_size};
    
    actConst_  = actConstant{convConst1_.out_batch, convConst1_.out_slice,
        convConst1_.out_height, convConst1_.out_width
    };
    
    convConst2_ = conv::makeConvConstant(outShape1_, conv2_.getParams());
    bnConst2_ = bnConstant {convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width, convConst2_.out_size};
    
    outShape1_ = shape{(uint)convConst1_.out_batch, (uint)conv1_.getParams().outC, (uint)convConst1_.out_height, (uint)convConst1_.out_width};
    outShape2_ = shape {(uint)convConst2_.out_batch, (uint)conv2_.getParams().outC, (uint)convConst2_.out_height, (uint)convConst2_.out_width};
}


id<MTLCommandBuffer> basicBlock::forward(id<MTLBuffer> input,shape& inShape, id<MTLCommandBuffer>* commandBufferP) {
    
    id<MTLCommandBuffer> commandBuffer = commandBufferP? *commandBufferP: [resource_->getCommandQueue() commandBuffer];
    
    makingConstantAndShape(inShape);
    
    
    // encode conv1
    id <MTLComputeCommandEncoder> convCommandEncoder1 = [commandBuffer computeCommandEncoder];
    [convCommandEncoder1 setComputePipelineState:conv1_.getPSO()];
    
    id<MTLBuffer> interBuffer1 = resource_->getBuffer(convConst1_.out_batch * convConst1_.out_slice * 4 * convConst1_.out_size);
    std::vector<id<MTLBuffer>> inOutBuffers {input, interBuffer1};
    conv1_.setBuffer(inOutBuffers, convCommandEncoder1);
    conv1_.setConstant(&convConst1_, convCommandEncoder1);
    
    MTLSize threadGroupCounts = MTLSizeMake(1, 1, 1);
    MTLSize threadgroups = MTLSizeMake(convConst1_.out_width , convConst1_.out_height,  (convConst1_.out_slice * convConst1_.out_batch));
    [convCommandEncoder1 dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [convCommandEncoder1 endEncoding];
    
    
    // encode bn1
    id<MTLComputeCommandEncoder> bnCommandEncoder1 = [commandBuffer computeCommandEncoder];
    [bnCommandEncoder1 setComputePipelineState:bn1_.getPSO()];
    inOutBuffers = {interBuffer1, interBuffer1};
    bn1_.setBuffer(inOutBuffers, bnCommandEncoder1);
    bn1_.setConstant(&bnConst1_, bnCommandEncoder1);
    threadGroupCounts = MTLSizeMake(1, 1, 1);
    threadgroups = MTLSizeMake(bnConst1_.batch * bnConst1_.slice * bnConst1_.height * bnConst1_.width, 1, 1);
    [bnCommandEncoder1 dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [bnCommandEncoder1 endEncoding];
    
    
    //encode relu
    id<MTLComputeCommandEncoder> reluCommandEncoder = [commandBuffer computeCommandEncoder];
    [reluCommandEncoder setComputePipelineState:relu_.getPSO()];
    inOutBuffers = {interBuffer1, interBuffer1};
    relu_.setBuffer(inOutBuffers, reluCommandEncoder);
    relu_.setConstant(&actConst_, reluCommandEncoder);
    threadGroupCounts = MTLSizeMake(1, 1, 1);
    threadgroups = MTLSizeMake(actConst_.batch * actConst_.slice * actConst_.height * actConst_.width, 1, 1);
    [reluCommandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    
    
    // encode conv2
    id <MTLComputeCommandEncoder> convCommandEncoder2 = [commandBuffer computeCommandEncoder];
    [convCommandEncoder2 setComputePipelineState:conv2_.getPSO()];
    id<MTLBuffer> interBuffer2 = resource_->getBuffer(convConst2_.out_batch * convConst2_.out_slice * 4 * convConst2_.out_size);
    inOutBuffers = {interBuffer1, interBuffer2};
    conv2_.setBuffer(inOutBuffers, convCommandEncoder2);
    conv2_.setConstant(&convConst2_, convCommandEncoder2);
    
    threadGroupCounts = MTLSizeMake(1, 1, 1);
    threadgroups = MTLSizeMake(convConst2_.out_width , convConst2_.out_height,  (convConst2_.out_slice * convConst2_.out_batch));
    [convCommandEncoder2 dispatchThreadgroups:threadgroups threadsPerThreadgroup: threadGroupCounts];
    [convCommandEncoder2 endEncoding];
    
    // encode bn2
    id<MTLComputeCommandEncoder> bnCommandEncoder2 = [commandBuffer computeCommandEncoder];
    [bnCommandEncoder2 setComputePipelineState:bn2_.getPSO()];
    inOutBuffers = {interBuffer2, interBuffer2};
    bn2_.setBuffer(inOutBuffers, bnCommandEncoder2);
    bn2_.setConstant(&bnConst2_, bnCommandEncoder2);
    threadGroupCounts = MTLSizeMake(1, 1, 1);
    threadgroups = MTLSizeMake(bnConst2_.batch * bnConst2_.slice * bnConst2_.height * bnConst2_.width, 1, 1);
    [bnCommandEncoder2  dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [bnCommandEncoder2 endEncoding];
    
    
    return commandBuffer;
    
    
}


//basicBlock makingBasicBlock(NSDictionary *infoFromJson) {
//    NSDictionary* conv1Info = infoFromJson[@"conv1"];
//    
//}
