//
//  interp.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;




kernel void interpBilinear(const device float4* in [[buffer(0)]],
                           device float4* out [[buffer(1)]],
                           constant interpConstant& cp [[buffer(2)]],
                           uint2 idx[[thread_position_in_grid]]){
    
    if((int)idx.x >= cp.out_width || (int)idx.y >= cp.out_height) return;
    float x_f = (float) idx.x / (cp.out_width - 1);
    float y_f = (float) idx.y / (cp.out_height -1);
    
    float x_in = x_f * cp.in_width - 0.5;
    int x_in_left = int(x_in);
    float x_frac = x_in - floor(x_in);
    
    float y_in = y_f * cp.in_height - 0.5;
    int y_in_top = int(y_in);
    float y_frac = y_in - floor(y_in);
    
    
}
