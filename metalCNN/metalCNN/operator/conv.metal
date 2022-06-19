//
//  conv.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include <metal_stdlib>
using namespace metal;


struct convParams{
    uint in_batch;
    uint in_slice; // slice per batch
    uint in_size; // size per slice, h*w
    uint in_height;
    uint in_width;
    
    uint out_batch;
    uint out_slice; // slice per batch
    uint out_size; //size per slice, h*w
    uint out_height;
    uint out_width;
    
    uint kernel_h;
    uint kernel_w;
    uint kernel_size;
    uint stride_h;
    uint stride_w;
    uint pad_x;
    uint pad_y;
};

kernel void conv(const device float4* in[[buffer(0)]],
                 device float4* out[[buffer(1)]],
                 const device float4x4* weight [[buffer(2)]],
                 const device float4* bias[[buffer(3)]],
                 constant convParams& cp [[buffer(4)]],
                 uint3 idx [[thread_position_in_grid]]
                 )
{
    if(idx.x >= cp.out_width || idx.y > cp.out_height ||
       idx.z >= (cp.out_slice * cp.out_batch)) {return;}

    uint idx_b = idx.z / cp.out_slice;
    uint idx_s = idx.z % cp.out_slice;

//    (cp.kernel_w / 2) + idx.x * cp.stride_w -
    uint in_start_x = idx.x * cp.stride_w - cp.pad_x;
    uint in_start_y = idx.y * cp.stride_h - cp.pad_y;
    uint in_start_x_clip = max((int)in_start_x, 0);
    uint in_start_y_clip = max((int)in_start_y, 0);
    uint in_end_x_clip = min(cp.in_width, in_start_x + cp.kernel_w);
    uint in_end_y_clip = min(cp.in_height, in_start_y + cp.kernel_h);
    
    uint kernelW = in_end_x_clip - in_start_x_clip;
    uint kernelH = in_end_y_clip - in_start_y_clip;

    const device float4* in_start_p = in + idx_b * cp.in_slice * cp.in_size + in_start_y_clip * cp.in_width + in_start_x_clip;
    
    const device float4x4* weight_start_p = weight + idx_s * cp.kernel_size * cp.in_slice + (in_start_y_clip - in_start_y) * cp.kernel_w + (in_start_x_clip - in_start_x);
    
    device float4* out_start_p = out + idx_b * cp.out_slice * cp.out_size + idx_s * cp.out_size + idx.y * cp.out_width + idx.x;

    float4 res = {0.0,0.0,0.0,0.0};
    for(uint i_s = 0; i_s < cp.in_slice; ++i_s) {
        for(uint i_h = 0; i_h < kernelH; ++i_h){
            for(uint i_w = 0; i_w < kernelW; ++i_w){
                float4x4 weight_data = weight_start_p[i_s * cp.kernel_size + i_h * cp.kernel_w + i_w];
                float4 in_data = in_start_p[i_s * cp.in_size + i_h * cp.in_width + i_w];
                res += (in_data * weight_data);

            }
        }
    }
    *out_start_p  = res;
}


