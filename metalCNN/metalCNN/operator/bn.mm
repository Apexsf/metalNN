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
        dst_p[i] = std::sqrt(src_p[i] + 0.00001) ;
    }
//    memcpy(runningVar_.contents, src_p, channel_);
    
}

void bn::setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBuffer:inOutBuffers[0] offset:0 atIndex:0];
    [commandEncoder setBuffer:inOutBuffers[1] offset:0 atIndex:1];
    [commandEncoder setBuffer:gamma_ offset:0 atIndex:2];
    [commandEncoder setBuffer:beta_ offset:0 atIndex:3];
    [commandEncoder setBuffer:runningMean_ offset:0 atIndex:4];
    [commandEncoder setBuffer:runningVarSqrtWithEps_ offset:0 atIndex:5];
}

void bn::setConstant(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBytes:constantP length:sizeof(bnConstant) atIndex:6];
}

void bn::dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    bnConstant* p = (bnConstant*) constantP;
    MTLSize threadGroupCounts = MTLSizeMake(1024, 1, 1);
    MTLSize threadgroups = MTLSizeMake(
                                      divUp(p->batch * p->slice * p->height * p->width,1024),
                                       1, 1);
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
}


