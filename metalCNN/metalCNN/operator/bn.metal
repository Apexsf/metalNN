//
//  bn.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/23.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;


kernel void bn(const device float4* in[[buffer(0)]],
               device float4* out [[buffer(1)]],
               const device float4* gamma [[buffer(2)]],
               const device float4* beta [[buffer(3)]],
               const device float4* running_mean [[buffer(4)]],
               const device float4* running_var_sqrt_with_eps [[buffer(5)]],
               constant bnConstant& cp [[buffer(6)]],
               uint idx [[thread_position_in_grid]]
               )

{
    if ((int)idx >= (cp.batch * cp.slice * cp.size)) return;
    int idx_slice = (idx / cp.size) % cp.slice;
    out[idx] = (in[idx] - running_mean[idx_slice]) /  running_var_sqrt_with_eps[idx_slice] * gamma[idx_slice] + beta[idx_slice];
//    out[idx] = in[idx];
}
