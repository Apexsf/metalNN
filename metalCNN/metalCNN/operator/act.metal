//
//  act.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;


kernel void relu (const device float4* in [[buffer(0)]],
                  device float4* out [[buffer(1)]],
                  constant reluConstant& cp [[buffer(2)]],
                  uint idx [[thread_position_in_grid]]) {
    if ((int)idx >= (cp.batch * cp.slice * cp.width * cp.height)) return;
    out[idx] = max(in[idx], 0);
}
