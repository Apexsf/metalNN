//
//  elemWise.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/26.
//

#include "elemWise.h"

elemWise::elemWise (std::shared_ptr<gpuResource> resource, std::string elemWiseOp) :
op (resource, elemWiseOp), elemWiseOp_(elemWiseOp){
    
}


void elemWise::setBuffer(std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBuffer:inOutBuffers[0] offset:0 atIndex:0];
    [commandEncoder setBuffer:inOutBuffers[1] offset:0 atIndex:1];
    [commandEncoder setBuffer:inOutBuffers[2] offset:0 atIndex:2];
}



void elemWise::setConstant(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBytes:constantP length:sizeof(elemWiseConstant) atIndex:3];
}


void elemWise::dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    elemWiseConstant* p = (elemWiseConstant*) constantP;
    MTLSize threadGroupCounts = MTLSizeMake(1024, 1, 1);
    MTLSize threadgroups = MTLSizeMake(
                                      divUp((p->batch * p->slice * p->height * p->width),1024),
                                       1, 1);
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
}
