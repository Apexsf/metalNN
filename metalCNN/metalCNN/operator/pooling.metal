//
//  pooling.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;


kernel void poolingMax (const device float4* in [[buffer(0)]],
                     device float4* out[[buffer(1)]],
                     constant poolingConstant& cp [[buffer(2)]],
                     uint3 idx [[thread_position_in_grid]]
                     ) {
    if((int)idx.x >= cp.out_width || (int)idx.y > cp.out_height ||
       (int)idx.z >= (cp.out_slice * cp.out_batch)) {return;}
    
    int in_start_x = idx.x * cp.stride_w - cp.pad_x;
    int in_start_y = idx.y * cp.stride_h - cp.pad_y;
    int in_start_x_clip = max((int)in_start_x, 0);
    int in_start_y_clip = max((int)in_start_y, 0);
    int in_end_x_clip = min(cp.in_width, in_start_x + cp.kernel_w);
    int in_end_y_clip = min(cp.in_height, in_start_y + cp.kernel_h);
    in_end_x_clip = max(in_end_x_clip, 0); // in case padX is too big
    in_end_y_clip = max(in_end_y_clip, 0); // in case padY is too big
    
    int kernelW = in_end_x_clip - in_start_x_clip;
    int kernelH = in_end_y_clip - in_start_y_clip;
    
    const device float4* in_start_p = in + idx.z * cp.in_size + in_start_y_clip * cp.in_width + in_start_x_clip;
    device float4* out_start_p = out + idx.z * cp.out_size + idx.y * cp.out_width + idx.x;
    
    float4 res = *in_start_p;
    for (int i_h = 0; i_h < kernelH; ++i_h) {
        for(int i_w = 0; i_w < kernelW; ++i_w){
            res = max(res, in_start_p[i_h * cp.in_width + i_w]);
        }
    }
    *out_start_p = res;
//    *out_start_p = *in_start_p;
}




kernel void poolingAvg (const device float4* in [[buffer(0)]],
                     device float4* out[[buffer(1)]],
                     constant poolingConstant& cp [[buffer(2)]],
                     uint3 idx [[thread_position_in_grid]]
                     ) {
    if((int)idx.x >= cp.out_width || (int)idx.y > cp.out_height ||
       (int)idx.z >= (cp.out_slice * cp.out_batch)) {return;}
    
    int in_start_x = idx.x * cp.stride_w - cp.pad_x;
    int in_start_y = idx.y * cp.stride_h - cp.pad_y;
    int in_start_x_clip = max((int)in_start_x, 0);
    int in_start_y_clip = max((int)in_start_y, 0);
    int in_end_x_clip = min(cp.in_width, in_start_x + cp.kernel_w);
    int in_end_y_clip = min(cp.in_height, in_start_y + cp.kernel_h);
    in_end_x_clip = max(in_end_x_clip, 0); // in case padX is too big
    in_end_y_clip = max(in_end_y_clip, 0); // in case padY is too big
    
    int kernelW = in_end_x_clip - in_start_x_clip;
    int kernelH = in_end_y_clip - in_start_y_clip;
    
    const device float4* in_start_p = in + idx.z * cp.in_size + in_start_y_clip * cp.in_width + in_start_x_clip;
    device float4* out_start_p = out + idx.z * cp.out_size + idx.y * cp.out_width + idx.x;
    
    float4 res = {0,0,0,0};
    for (int i_h = 0; i_h < kernelH; ++i_h) {
        for(int i_w = 0; i_w < kernelW; ++i_w){
            res += in_start_p[i_h * cp.in_width + i_w];
        }
    }
    *out_start_p = res / cp.kernel_size;
//    *out_start_p = *in_start_p;
}
