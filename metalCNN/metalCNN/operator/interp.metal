//
//  interp.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;


// see: https://stackoverflow.com/questions/70024313/resize-using-bilinear-interpolation-in-python  for detailed bilinear interpolation implementation
kernel void interpBilinear(const device float4* in [[buffer(0)]],
                           device float4* out[[buffer(1)]],
                           constant interpBilinearConstant& cp [[buffer(2)]],
                           uint3 idx [[thread_position_in_grid]]
                           ) {
    if( (int)idx.x >=  cp.out_width || (int) idx.y >= cp.out_height
       || (int)idx.z >= (cp.out_batch * cp.out_slice)) return;
    
    float in_x = idx.x * cp.w_scale + cp.w_offset;
    float in_y = idx.y * cp.h_scale + cp.h_offset;
    
    int in_left = int(floor(in_x));
    int in_top = int(floor(in_y));
    int in_right = int(ceil(in_x));
    int in_bottom = int(ceil(in_y));
    
    in_left = clamp(in_left, 0, cp.in_width - 1);
    in_right = clamp(in_right   , 0, cp.in_width - 1);
    in_top = clamp(in_top, 0, cp.in_height - 1);
    in_bottom = clamp(in_bottom, 0, cp.in_height-1);
    
    float in_x_factor = in_x - in_left;
    float in_y_factor = in_y - in_top;
    
    const device float4* in_top_p = in + idx.z * cp.in_size
    + in_top * cp.in_width;
    const device float4* in_bottom_p = in + idx.z * cp.in_size +
    in_bottom * cp.in_width;
    
    
    float4 n1 = in_top_p[in_left] * (1-in_x_factor) * (1-in_y_factor);
    float4 n2 = in_top_p[in_right]* in_x_factor * (1 - in_y_factor);
    float4 n3 =in_bottom_p[in_left]* (1-in_x_factor) * in_y_factor;
    float4 n4 = in_bottom_p[in_right] * in_x_factor * in_y_factor;
    
    
    out[idx.z * cp.out_size + idx.y * cp.out_width + idx.x] =
    (n1 + n2 + n3 + n4);
    

}
