//
//  conv.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include "conv.h"

conv::conv(std::shared_ptr<gpuResource> resource, std::string name, const convParams& params) : op(resource, name), params_(params){
    
}

void conv::loadWeight(const tensor& t){
    uint ic = params_.inC;
    uint oc = params_.outC;
    uint h = params_.kernelH;
    uint w  = params_.kernelW;
    
    uint icDU4 = divUp(ic, 4);
    uint ocDU4 = divUp(oc, 4);
    
//    uint ocRU4 = roundUp(ocDU4, 4);
//    size_t dstSize = ocRU4 * icDU4 * 16 * h * w;
    size_t dstSize = ocDU4 * icDU4 * 16 * h * w;
    
    weight_ = [getResource()->getDevice() newBufferWithLength:dstSize * sizeof(float) options:MTLResourceStorageModeShared];
    
    float* src_p = t.getRawPointer();
    float* dst_p = (float*) weight_.contents;
    
    for(uint i_oc = 0; i_oc< oc; ++i_oc){
        uint dst_o_order = i_oc  / 4;
        uint dst_o_remainder = i_oc  % 4;
        float* dst_o_p = dst_p + dst_o_order * icDU4 * 16 * h * w + dst_o_remainder * 4;
        for(uint i_ic = 0; i_ic < ic; ++i_ic) {
            uint dst_i_order = i_ic  / 4;
            uint dst_i_remainder = i_ic  % 4;
            float* dst_i_p = dst_o_p + dst_i_order * 16 * h * w + dst_i_remainder;
            for(uint i_h = 0; i_h < h; i_h++){
                for(uint i_w = 0; i_w < w; i_w++){
                    dst_i_p[(i_h * w + i_w) * 16] = *src_p;
                    src_p++;
                }
            }
        }
    }
    
    std::cout<<'d';
}


void conv::execute(id<MTLBuffer> input, id<MTLBuffer> output, const convRunTimeConstant& constant){
    id <MTLCommandBuffer> commandBuffer = [getResource()->getCommandQueue() commandBuffer];
    id <MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:getPSO()];
    
    [commandEncoder setBuffer:input offset:0 atIndex:0];
    [commandEncoder setBuffer:output offset:0 atIndex:1];
    [commandEncoder setBuffer:weight_ offset:0 atIndex:2];
    [commandEncoder setBuffer:bias_ offset:0 atIndex:3];
    [commandEncoder setBytes:&constant length:sizeof(convRunTimeConstant) atIndex:4];
    
    
    MTLSize threadGroupCounts = MTLSizeMake(4, 4, 4);
    MTLSize threadgroups = MTLSizeMake( 8 , 8,  1);
    
    
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [commandEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}
