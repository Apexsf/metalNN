//
//  act.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#include "act.h"


act::act(std::shared_ptr<gpuResource> resource, std::string act_name): op(resource, act_name), actName_(act_name) {
    
}


void act::setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) {
    [commandEncoder setBuffer:inOutBuffers[0] offset:0 atIndex:0];
    [commandEncoder setBuffer:inOutBuffers[1] offset:0 atIndex:1];
}

void act::setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) {
    if (actName_ == "relu") {
        reluConstant* p = (reluConstant*) constantP;
        [commandEncoder setBytes:p length:sizeof(reluConstant) atIndex:2];
    } else {
        std::cerr << "currently only support relu activation" << std::endl;
        abort();
    }
}


void act::dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) {
    reluConstant* p = (reluConstant*) constantP;
    MTLSize threadGroupCounts = MTLSizeMake(1, 1, 1);
    MTLSize threadgroups = MTLSizeMake(p->batch * p->slice * p->height * p->width, 1, 1);
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
}

