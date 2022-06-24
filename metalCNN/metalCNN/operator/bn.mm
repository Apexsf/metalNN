//
//  bn.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/21.
//

#include "bn.h"

bn::bn(std::shared_ptr<gpuResource> resource, std::string name, uint channel) :  op(resource, name) , channel_(channel){

}

void bn::loadWeight(std::map<std::string, tensor>& weights){
    int channelRD = roundUp(channel_, 4);
    gamma_ = makingBuffer(channelRD * sizeof(float), MTLResourceStorageModeShared);
    beta_ = makingBuffer(channelRD * sizeof(float), MTLResourceStorageModeShared);
    runningMean_ = makingBuffer(channelRD * sizeof(float), MTLResourceStorageModeShared);
    runningVarSqrtWithEps_ = makingBuffer(channelRD * sizeof(float), MTLResourceStorageModeShared);
    
    float* src_p;
    
    src_p = weights["gamma"].getRawPointer();
    memcpy(gamma_.contents,src_p, channel_ * sizeof(float));
    
    src_p = weights["beta"].getRawPointer();
    memcpy(beta_.contents, src_p, channel_ * sizeof(float));
    
    src_p = weights["running_mean"].getRawPointer();
    memcpy(runningMean_.contents, src_p, channel_ * sizeof(float));
    
    src_p = weights["running_var"].getRawPointer();
    float* dst_p = (float*)runningVarSqrtWithEps_.contents;
    for(uint i = 0; i < channel_; ++i){
        dst_p[i] = std::sqrt(src_p[i]) + 0.00001;
    }
//    memcpy(runningVar_.contents, src_p, channel_);
    
}


void bn::execute(id<MTLBuffer> input, id <MTLBuffer> output, const shape& shp) {
    id <MTLCommandBuffer> commandBuffer = [getResource()->getCommandQueue() commandBuffer];
    id <MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:getPSO()];
    
    bnConstant cp{(int)shp.batch, (int)divUp(shp.channel, 4), (int)shp.width * (int)shp.height};
    
    [commandEncoder setBuffer:input offset:0 atIndex:0];
    [commandEncoder setBuffer:output offset:0 atIndex:1];
    [commandEncoder setBuffer:gamma_ offset:0 atIndex:2];
    [commandEncoder setBuffer:beta_ offset:0 atIndex:3];
    [commandEncoder setBuffer:runningMean_ offset:0 atIndex:4];
    [commandEncoder setBuffer:runningVarSqrtWithEps_ offset:0 atIndex:5];
    [commandEncoder setBytes:&cp length:sizeof(bnConstant) atIndex:6];
    
    
    MTLSize threadGroupCounts = MTLSizeMake(1, 1, 1);
    MTLSize threadgroups = MTLSizeMake(shp.batch * divUp(shp.channel, 4) * shp.width * shp.height, 1, 1);
    
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
}
