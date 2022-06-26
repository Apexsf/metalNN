//
//  elemWise.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/26.
//

#ifndef elemWise_h
#define elemWise_h

#include "operator.h"
#include "metalConstant.metal"


class elemWise: public op {
public:
    elemWise (std::shared_ptr<gpuResource> resource, std::string elemWiseOp);
    
    virtual void loadWeight(std::map<std::string, tensor>& weights) override {
        
    }
    
    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) override;
    

private:
    std::string elemWiseOp_;
    
};


#endif /* elemWise_hpp */
