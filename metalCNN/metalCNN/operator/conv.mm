//
//  conv.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include "conv.h"

conv::conv(std::shared_ptr<gpuResource> resource, std::string name, const convParams& params) : op(resource, name), params_(params){
    
}

void conv::loadWeight(std::map<std::string, tensor>& weights){
    uint ic = params_.inC;
    uint oc = params_.outC;
    uint h = params_.kernelH;
    uint w  = params_.kernelW;
    
    uint icDU4 = divUp(ic, 4);
    uint ocDU4 = divUp(oc, 4);
    
//    uint ocRU4 = roundUp(ocDU4, 4);
//    size_t dstSize = ocRU4 * icDU4 * 16 * h * w;
    size_t dstSize = ocDU4 * icDU4 * 16 * h * w;
    
    weight_ = makingBuffer(dstSize * sizeof(float), MTLResourceStorageModeShared);
    
    float* src_p = weights["weight"].getRawPointer();
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
    
    // load bias
    uint outCDU = roundUp(params_.outC, 4);
    bias_ = makingBuffer(outCDU * sizeof(float), MTLResourceStorageModeShared);
    float* dst_b_p = (float*) bias_.contents;
    if (weights.find("bias") != weights.end()) {
        float* src_b_p = weights["bias"].getRawPointer();
        uint c = weights["bias"].getShape().channel;
        for(uint i = 0; i < c; i++){
            dst_b_p[i] = src_b_p[i];
        }
        for(uint i = c; i < outCDU; i++){
            dst_b_p[i] = 0;
        }
    }else {
        for(uint i = 0; i < outCDU; i++){
            dst_b_p[i] = 0.0;
        }
    }
}

void conv::setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBuffer:inOutBuffers[0] offset:0 atIndex:0];
    [commandEncoder setBuffer:inOutBuffers[1] offset:0 atIndex:1];
    [commandEncoder setBuffer:weight_ offset:0 atIndex:2];
    [commandEncoder setBuffer:bias_ offset:0 atIndex:3];
}
void conv::setConstant(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    [commandEncoder setBytes:constantP length:sizeof(convConstant) atIndex:4];
}

void conv::dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    convConstant* p = (convConstant*) constantP;
    MTLSize threadGroupCounts = MTLSizeMake(1, 1, 1);
    MTLSize threadgroups = MTLSizeMake(p->out_width , p->out_height,  (p->out_slice * p->out_batch));
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
}




//void conv::execute(id<MTLBuffer> input, id<MTLBuffer> output, const convConstant& constant){
//    id <MTLCommandBuffer> commandBuffer = [getResource()->getCommandQueue() commandBuffer];
//    id <MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//    [commandEncoder setComputePipelineState:getPSO()];
//
//    [commandEncoder setBuffer:input offset:0 atIndex:0];
//    [commandEncoder setBuffer:output offset:0 atIndex:1];
//    [commandEncoder setBuffer:weight_ offset:0 atIndex:2];
//    [commandEncoder setBuffer:bias_ offset:0 atIndex:3];
//    [commandEncoder setBytes:&constant length:sizeof(convConstant) atIndex:4];
//
//
//    MTLSize threadGroupCounts = MTLSizeMake(1, 1, 1);
//    MTLSize threadgroups = MTLSizeMake(constant.out_width , constant.out_height,  (constant.out_slice * constant.out_batch));
//
//
//    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
//    [commandEncoder endEncoding];
//    [commandBuffer commit];
//    [commandBuffer waitUntilCompleted];
//}

shape conv::calOutShape(const shape& inShape, const convParams& params){
    shape outShape;
    outShape.batch = inShape.batch;
    outShape.channel = params.outC;
    outShape.height = (inShape.height - params.kernelH + 2 * params.padH) / params.strideH + 1;
    outShape.width = (inShape.width - params.kernelW + 2 *
                      params.padW) / params.strideW + 1;
    return outShape;
}

convConstant conv::makeConvConstant(const shape& inShape, const convParams& params){
    convConstant constant;
    shape outShape = calOutShape(inShape, params);
    
    constant.in_batch = inShape.batch;
    constant.in_slice = divUp(inShape.channel, 4);
    constant.in_size = inShape.width * inShape.height;
    constant.in_height = inShape.height;
    constant.in_width = inShape.width;
    
    constant.out_batch = inShape.batch;
    constant.out_slice = divUp(params.outC, 4);
    constant.out_size = outShape.width * outShape.height;
    constant.out_height = outShape.height;
    constant.out_width = outShape.width;
    
    constant.kernel_h = params.kernelH;
    constant.kernel_w = params.kernelW;
    constant.kernel_size = params.kernelW * params.kernelH;
    constant.stride_h = params.strideH;
    constant.stride_w = params.strideW;
    constant.pad_x = params.padW;
    constant.pad_y = params.padH;
    
    return constant;
    
}
