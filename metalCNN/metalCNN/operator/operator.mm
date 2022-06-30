//
//  operator.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include "operator.h"

id<MTLBuffer> op:: makingBuffer(size_t byteSize, int mode){
    id<MTLBuffer> buffer = [resource_->getDevice() newBufferWithLength:byteSize options:mode];
    return buffer;
}



void op::encodeCommand(std::vector<id<MTLBuffer>>& inOutBuffers, void* constantP,
                       id<MTLCommandBuffer> commandBuffer) {
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:getPSO()];
    setBuffer(inOutBuffers, commandEncoder);
    setConstant(constantP, commandEncoder);
    dispatch(constantP, commandEncoder);
    [commandEncoder endEncoding];
}


void op::runOnce(std::vector<id<MTLBuffer>>& inOutBuffers, void* constantP){
    id<MTLCommandBuffer> commandBuffer = newCommandBuffer();
    encodeCommand(inOutBuffers, constantP, commandBuffer);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}
