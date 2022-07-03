//
//  matmul.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;      


kernel void matmul(const device float4* in [[buffer(0)]],
             device float* out[[buffer(1)]],
             const device float4* weight [[buffer(2)]],
             const device float* bias [[buffer(3)]],
            constant matmulConstant& cp [[buffer(4)]],
            uint idx [[thread_position_in_grid]]
                   )
{
    if((int)idx  >= (cp.batch * cp.outC)) return;
    int idx_batch = idx / cp.outC;
    int idx_C = idx % cp.outC;
    
    const device float4* weight_start_p = weight + idx_C * cp.inSlice;
    const device float4* in_start_p = in + idx_batch * cp.inSlice;
    
    
    
    float4 res = {0.0, 0.0, 0.0, 0.0};
    for(int i = 0; i < cp.inSlice; ++i){
        res += weight_start_p[i] * in_start_p[i];
    }
    out[idx] = res[0] + res[1] + res[2] + res[3] + bias[idx_C];
}
