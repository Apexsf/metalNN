//
//  pooling.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#include "pooling.h"

pooling::pooling(std::shared_ptr<gpuResource> resource, std::string poolingMode, const poolingParams& params): op(resource, poolingMode), poolingMode_(poolingMode),params_(params){
    
}


void pooling::setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBuffer:inOutBuffers[0] offset:0 atIndex:0];
    [commandEncoder setBuffer:inOutBuffers[1] offset:0 atIndex:1];
}
void pooling::setConstant(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBytes:constantP length:sizeof(poolingConstant) atIndex:2];
}

void pooling::dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    poolingConstant* p = (poolingConstant*) constantP;
    MTLSize threadGroupCounts = MTLSizeMake(1, 1, 1);
    MTLSize threadgroups = MTLSizeMake(p->out_width , p->out_height,  (p->out_slice * p->out_batch));
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
}


poolingConstant pooling::makePoolingConstant(const shape& inShape, const poolingParams& params){
    poolingConstant constant;
    
    constant.in_batch = inShape.batch;
    constant.in_slice = divUp(inShape.channel, 4);
    constant.in_size = inShape.width * inShape.height;
    constant.in_height = inShape.height;
    constant.in_width = inShape.width;
    
    int outH = (inShape.height - params.kernelH + 2 * params.padH) / params.strideH + 1;
    int outW = (inShape.width - params.kernelW + 2 * params.padW) / params.strideW + 1;
    
    constant.out_batch = inShape.batch;
    constant.out_slice = divUp(inShape.channel, 4);
    constant.out_size = outH * outW;
    constant.out_height = outH;
    constant.out_width = outW;
    
    constant.kernel_h = params.kernelH;
    constant.kernel_w = params.kernelW;
    constant.kernel_size = params.kernelW * params.kernelH;
    constant.stride_h = params.strideH;
    constant.stride_w = params.strideW;
    constant.pad_x = params.padW;
    constant.pad_y = params.padH;
    
    return constant;
}
