//
//  convBnRelu.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/26.
//

#ifndef convBnRelu_h
#define convBnRelu_h
#include <memory>
#include "gpuResource.h"
#include "conv.h"
#include "bn.h"
#include "act.h"

class convBnRelu {
public:
    convBnRelu(std::shared_ptr<gpuResource> resource, convParams& convPara, uint
               bnChannel);
    void run(id <MTLBuffer> input, id<MTLBuffer> output1, id<MTLBuffer> output2,
             convConstant& convConst, bnConstant& bnConst, actConstant& actConst);


public:
    std::shared_ptr<gpuResource> resource_;
    conv conv_;
    bn bn_;
    act relu_;
//    id <MTLFunction> convShader_;
//    id <MTLFunction> bnShader_;
//    id <MTLFunction> reluShader_;
//
//    id <MTLComputePipelineState> convPSO_;
//    id <MTLComputePipelineState> bnPSO_;
//    id <MTLComputePipelineState> reluPSO_;
};


#endif /* convBnRelu_hpp */
