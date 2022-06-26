//
//  convBnRelu.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/26.
//

#include "convBnRelu.h"

convBnRelu::convBnRelu(std::shared_ptr<gpuResource> resource, convParams& convPara,
                       uint bnChannel): resource_(resource), conv_(resource, "conv", convPara), bn_(resource, "bn", bnChannel), relu_(resource, "relu") {
//    convShader_ = [resource->getLibrary() newFunctionWithName:@"conv"];
//    bnShader_ = [resource->getLibrary() newFunctionWithName:@"bn"];
//    reluShader_ = [resource->getLibrary() newFunctionWithName:@"relu"];
//
//    convPSO_ = [resource->getDevice() newComputePipelineStateWithFunction:convShader_ error:NULL];
//    bnPSO_ = [resource->getDevice() newComputePipelineStateWithFunction:bnShader_ error:NULL];
//    reluPSO_ = [resource->getDevice() newComputePipelineStateWithFunction:reluShader_ error:NULL];
    
}


void convBnRelu::run(id <MTLBuffer> input, id<MTLBuffer> output1, id<MTLBuffer> output2,
                     convConstant& convConst, bnConstant& bnConst, actConstant& actConst) {
    id <MTLCommandBuffer> commandBuffer =  [resource_->getCommandQueue() commandBuffer];

    id <MTLComputeCommandEncoder> convCommandEncoder = [commandBuffer computeCommandEncoder];
    [convCommandEncoder setComputePipelineState:conv_.getPSO()];
    [convCommandEncoder setBuffer:input offset:0 atIndex:0];
    [convCommandEncoder setBuffer:output1 offset:0 atIndex:1];
    [convCommandEncoder setBuffer:conv_.getWeight() offset:0 atIndex:2];
    [convCommandEncoder setBuffer:conv_.getBias() offset:0 atIndex:3];
    [convCommandEncoder setBytes:&convConst length:sizeof(convConstant) atIndex:4];
    
    MTLSize threadGroupCounts = MTLSizeMake(1, 1, 1);
    MTLSize threadgroups = MTLSizeMake(convConst.out_width , convConst.out_height,  (convConst.out_slice * convConst.out_batch));
    [convCommandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [convCommandEncoder endEncoding];

    
    id <MTLComputeCommandEncoder> bnCommandEncoder = [commandBuffer computeCommandEncoder];
    [bnCommandEncoder setComputePipelineState:bn_.getPSO()];
    [bnCommandEncoder setBuffer:output1 offset:0 atIndex:0];
    [bnCommandEncoder setBuffer:output2 offset:0 atIndex:1];
    [bnCommandEncoder setBuffer:bn_.getGamma() offset:0 atIndex:2];
    [bnCommandEncoder setBuffer:bn_.getBeta() offset:0 atIndex:3];
    [bnCommandEncoder setBuffer:bn_.getRunningMean() offset:0 atIndex:4];
    [bnCommandEncoder setBuffer:bn_.getRunningVarSqrtWithEps() offset:0 atIndex:5];
    [bnCommandEncoder setBytes:&bnConst length:sizeof(bnConst) atIndex:6];

    threadGroupCounts = MTLSizeMake(1, 1, 1);
    threadgroups = MTLSizeMake(bnConst.batch * bnConst.slice * bnConst.height * bnConst.width, 1, 1);
    [bnCommandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [bnCommandEncoder endEncoding];

    id <MTLComputeCommandEncoder> reluCommandEncoder = [commandBuffer computeCommandEncoder];
    [reluCommandEncoder setComputePipelineState:relu_.getPSO()];
    [reluCommandEncoder setBuffer:output2 offset:0 atIndex:0];
    [reluCommandEncoder setBuffer:output1 offset:0 atIndex:1];
    [reluCommandEncoder setBytes:&actConst length:sizeof(actConst) atIndex:2];
    threadGroupCounts = MTLSizeMake(1, 1, 1);
    threadgroups = MTLSizeMake(actConst.batch * actConst.slice * actConst.height * actConst.width, 1, 1);
    [reluCommandEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadGroupCounts];
    [reluCommandEncoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
}

