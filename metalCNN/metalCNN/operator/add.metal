//
//  add.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include <metal_stdlib>
using namespace metal;

kernel void add (device const float* in1 [[buffer(0)]],
                device const float* in2 [[buffer(1)]],
                 device float* out [[buffer(2)]],
                 uint index [[thread_position_in_grid]]) {
    float res = in1[index] + in2[index];
    out[index] = res;
}


