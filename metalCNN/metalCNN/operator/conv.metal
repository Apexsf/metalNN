//
//  conv.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;



kernel void conv(const device float4* in[[buffer(0)]],
                 device float4* out[[buffer(1)]],
                 const device float4x4* weight [[buffer(2)]],
                 const device float4* bias[[buffer(3)]],
                 constant convConstant& cp [[buffer(4)]],
                 uint3 idx [[thread_position_in_grid]]
                 )
{
    
    if((int)idx.x >= cp.out_width || (int)idx.y > cp.out_height ||
       (int)idx.z >= (cp.out_slice * cp.out_batch)) {return;}

    int idx_b = idx.z / cp.out_slice;
    int idx_s = idx.z % cp.out_slice;

//    (cp.kernel_w / 2) + idx.x * cp.stride_w -
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

    const device float4* in_start_p = in + idx_b * cp.in_slice * cp.in_size + in_start_y_clip * cp.in_width + in_start_x_clip;
    
    const device float4x4* weight_start_p = weight + idx_s * cp.kernel_size * cp.in_slice + (in_start_y_clip - in_start_y) * cp.kernel_w + (in_start_x_clip - in_start_x);
    
    device float4* out_start_p = out + idx_b * cp.out_slice * cp.out_size + idx_s * cp.out_size + idx.y * cp.out_width + idx.x;

    float4 res = bias[idx_s];
    for(int i_s = 0; i_s < cp.in_slice; ++i_s) {
        for(int i_h = 0; i_h < kernelH; ++i_h){
            for(int i_w = 0; i_w < kernelW; ++i_w){
                float4x4 weight_data = weight_start_p[i_s * cp.kernel_size + i_h * cp.kernel_w + i_w];
                float4 in_data = in_start_p[i_s * cp.in_size + i_h * cp.in_width + i_w];
                res += (in_data * weight_data);

            }
        }
    }
    *out_start_p  = res;
}


