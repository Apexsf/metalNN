//
//  matmul.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#ifndef matmul_h
#define matmul_h
#import <Metal/Metal.h>
#include <memory>
#include "gpuResource.h"
#include "operator.h"
#include "metalConstant.metal"



class matmul : public op {
public:
    matmul(std::shared_ptr<gpuResource> resource, uint inC, uint outC);
    
    uint getInC(){
        return inC_;
    }
    
    uint getOutC() {
        return outC_;
    }
    
    
    virtual void loadWeight(std::map<std::string, tensor>& weights) override;
    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) override;
    
    
    
private:
    uint inC_;
    uint outC_;
    id<MTLBuffer> weight_;
    id<MTLBuffer> bias_;
};


#endif /* matmul_h */
