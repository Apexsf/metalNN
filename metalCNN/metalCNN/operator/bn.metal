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
               const device float4* gamma,
               const device float4* beta,
               const device float4* running_mean,
               const device float4* running_var,
               device float4* out [[buffer(1)]],
               constant bnConstant& cp,
               uint idx [[thread_position_in_grid]]
               )

{
    if ((int)idx >= (cp.batch * cp.slice * cp.size)) return;
    int idx_slice = idx % cp.size;
    out[idx] = (in[idx] - running_mean[idx_slice]) /  running_var[idx_slice] * gamma[idx] + beta[idx];
    
}
