//
//  act.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#ifndef act_h
#define act_h
#include "operator.h"
#include "metalConstant.metal"

class act : public op {
public:
    act(std::shared_ptr<gpuResource> resource, std::string act_name);
    virtual void loadWeight(std::map<std::string, tensor>& weights) override {
        
    }
    
    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) override;

private:
    std::string actName_;
    

};


#endif /* act_hpp */
