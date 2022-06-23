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
    runningVar_ = makingBuffer(channelRD * sizeof(float), MTLResourceStorageModeShared);
    
    float* src_p;
    
    src_p = weights["gamma"].getRawPointer();
    memcpy(gamma_.contents,src_p, channel_);
    
    src_p = weights["beta"].getRawPointer();
    memcpy(beta_.contents, src_p, channel_);
    
    src_p = weights["runningMean"].getRawPointer();
    memcpy(runningMean_.contents, src_p, channel_);
    
    src_p = weights["var"].getRawPointer();
    memcpy(runningVar_.contents, src_p, channel_);
    
}


void execute(id<MTLBuffer> input, id <MTLBuffer> output) {
    
}
