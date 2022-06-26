//
//  elemWise.metal
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/26.
//

#include <metal_stdlib>
#include "metalConstant.metal"
using namespace metal;


kernel void elemWiseAdd(const device float4* in1 [[buffer(0)]],
                        const device float4* in2 [[buffer(1)]],
                        device float4* out[[buffer(2)]],
                        constant elemWiseConstant& cp[[buffer(3)]],
                        uint idx [[thread_position_in_grid]]) {
    if ((int)idx >= (cp.batch * cp.slice * cp.size)) return;
    out[idx] = in1[idx] + in2[idx];
}


kernel void elemWiseMul(const device float4* in1 [[buffer(0)]],
                        const device float4* in2 [[buffer(1)]],
                        device float4* out[[buffer(2)]],
                        constant elemWiseConstant& cp[[buffer(3)]],
                        uint idx [[thread_position_in_grid]]
                        ) {
    if ((int)idx >= (cp.batch * cp.slice * cp.size)) return;
    out[idx] = in1[idx] * in2[idx];
}


kernel void elemWiseSub(const device float4* in1 [[buffer(0)]],
                        const device float4* in2 [[buffer(1)]],
                        device float4* out[[buffer(2)]],
                        constant elemWiseConstant& cp[[buffer(3)]],
                        uint idx [[thread_position_in_grid]]
                        ) {
    if ((int)idx >= (cp.batch * cp.slice * cp.size)) return;
    out[idx] = in1[idx] - in2[idx];
}


kernel void elemWiseDiv(const device float4* in1 [[buffer(0)]],
                        const device float4* in2 [[buffer(1)]],
                        device float4* out[[buffer(2)]],
                        constant elemWiseConstant& cp[[buffer(3)]],
                        uint idx [[thread_position_in_grid]]
                        ) {
    if ((int)idx >= (cp.batch * cp.slice * cp.size)) return;
    out[idx] = in1[idx] / in2[idx];
}
