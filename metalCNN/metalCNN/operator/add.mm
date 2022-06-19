//
//  add.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include "add.h"

void add::execute() {
    id <MTLCommandBuffer> commandBuffer = [getResource()->getCommandQueue() commandBuffer];
    id <MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:getPSO()];
    
    [commandEncoder setBuffer:inBuffer1_ offset:0 atIndex:0];
    [commandEncoder setBuffer:inBuffer2_ offset:0 atIndex:1];
    [commandEncoder setBuffer:outBuffer_ offset:0 atIndex:2];
    
    int groupSize = 1024;
    MTLSize threadGroupCounts = MTLSizeMake(groupSize, 1, 1);
    MTLSize threadgroups = MTLSizeMake(5120, 1, 1);
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}
