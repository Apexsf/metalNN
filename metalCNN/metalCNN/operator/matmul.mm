//
//  matmul.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#include "matmul.h"


matmul::matmul(std::shared_ptr<gpuResource> resource, uint inC, uint outC):
op(resource, "matmul"), inC_(inC), outC_(outC){
    
}


void matmul:: loadWeight(std::map<std::string, tensor>& weights){
    uint icRU4 = roundUp(inC_, 4);
    size_t dstSize = icRU4 * outC_;
    weight_ = makingBuffer(dstSize*sizeof(float), MTLResourceStorageModeShared);
    
    //load weight
    float* src_p = weights["weight"].getRawPointer();
    float* dst_p = (float*) weight_.contents;
    
    for(uint i_o = 0; i_o < outC_; ++i_o){
        for(uint i_i = 0; i_i < icRU4; ++i_i){
            if (i_i < inC_){
                dst_p[i_o * icRU4 + i_i] = *src_p;
                src_p++;
            }
        }
    }
    
    // load bias or assign to zero
    bias_ = makingBuffer(outC_ * sizeof(float), MTLResourceStorageModeShared);
    dst_p = (float*) bias_.contents;
    if(weights.find("bias") != weights.end()){
        src_p = weights["bias"].getRawPointer();
        for(uint i = 0; i < outC_; ++i){
            dst_p[i] = src_p[i];
        }
    } else {
        for(uint i = 0; i <  outC_; ++i){
            dst_p[i] = 0.0;
        }
    }
}


void matmul::setBuffer(std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBuffer:inOutBuffers[0] offset:0 atIndex:0];
    [commandEncoder setBuffer:inOutBuffers[1] offset:0 atIndex:1];
    [commandEncoder setBuffer:weight_ offset:0 atIndex:2];
    [commandEncoder setBuffer:bias_ offset:0 atIndex:3];
}


void matmul::setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBytes:constantP length:sizeof(matmulConstant) atIndex:4];
}

void matmul:: dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    matmulConstant* p = (matmulConstant*) constantP;
    MTLSize threadGroupCounts = MTLSizeMake(1024, 1, 1);
    MTLSize threadgroups = MTLSizeMake(
                                       divUp(p->batch * p->outC, 1024),1,1
                                       );
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    
}
