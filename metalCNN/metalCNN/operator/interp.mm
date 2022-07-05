//
//  interp.c
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#include "interp.h"


interp::interp(std::shared_ptr<gpuResource> resource, std::string interpModeName,
               const interpParams& params): op(resource, interpModeName), params_(params) {

     if (interpModeName == "interpBilinear"){
        mode_ = interpMode::BILINEAR;
    } else {
        std::cerr << "not soppurt : " << interpModeName << std::endl;
        abort();
    }
}


void interp::setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) {
    [commandEncoder setBuffer:inOutBuffers[0] offset:0 atIndex:0];
    [commandEncoder setBuffer:inOutBuffers[1] offset:0 atIndex:1];
    
}

void interp::setConstant(void* constantP, id<MTLComputeCommandEncoder> commandEncoder){
    if (mode_ == interpMode::BILINEAR){
        [commandEncoder setBytes:constantP length:sizeof(interpBilinearConstant) atIndex:2];
    } else {
        abort();
    }
}


void interp::dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) {
    
    interpBilinearConstant* p = (interpBilinearConstant* )constantP;
    MTLSize threadGroupCounts = MTLSizeMake(16, 16, 4);
    MTLSize threadgroups = MTLSizeMake(divUp(p->out_width, 16), divUp(p->out_height, 16), divUp(p->out_slice * p->out_batch, 4)
                                       );
    [commandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    
}

interpBilinearConstant interp::makingBilinearConstant(const shape& inShape, const interpParams& params) {
    
    interpBilinearConstant constant;
    
    constant.in_batch = inShape.batch;
    constant.in_slice = divUp(inShape.channel, 4);
    constant.in_size = inShape.width * inShape.height;
    constant.in_height = inShape.height;
    constant.in_width = inShape.width;
    
    constant.out_batch = inShape.batch;
    constant.out_slice = divUp(inShape.channel, 4);
    
    if(params.mode == interpParams::targetMode::SIZE){
        constant.out_height = params.outH;
        constant.out_width = params.outW;
        constant.out_size = params.outH  * params.outW;
        
    } else {
        constant.out_width = std::round (params.scaleW * constant.in_width);
        constant.out_height = std::round( params.scaleH * constant.in_height);
        constant.out_size = constant.out_width * constant.out_height;
    }
    
    constant.w_scale = (float)constant.in_width / (float)constant.out_width;
    constant.h_scale = (float)constant.in_height / (float)constant.out_height;
    
    float w_ori_center = (constant.in_width - 1) / 2.0;
    float h_ori_center = (constant.in_height - 1) / 2.0;
    float w_scale_center = (constant.out_width - 1) / 2.0;
    float h_scale_center = (constant.out_height - 1) / 2.0;
    
    constant.w_offset = -w_scale_center * constant.w_scale + w_ori_center;
    constant.h_offset = - h_scale_center * constant.h_scale + h_ori_center;
    
    
    return constant;
    
}
