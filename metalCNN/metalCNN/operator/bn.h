//
//  bn.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/21.
//

#ifndef bn_h
#define bn_h
#include "operator.h"
#include "metalConstant.metal"

class bn : public op {
public:
    bn(std::shared_ptr<gpuResource> resource, std::string name, uint channel);
    const id<MTLBuffer> getGamma() const {return gamma_;}
    const id<MTLBuffer> getBeta() const {return beta_;}
    const id<MTLBuffer> getRunningMean() const {return runningMean_;}
    const id<MTLBuffer> getRunningVarSqrtWithEps() const {return runningVarSqrtWithEps_;}
    
    virtual void loadWeight(std::map<std::string, tensor>& weights) override;
    
//    void execute (id<MTLBuffer>input, id<MTLBuffer> output, const shape& shp);
    
    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) override;
    
    
private:
    uint channel_;
    id<MTLBuffer> gamma_;
    id<MTLBuffer> beta_;
    id<MTLBuffer> runningMean_;
    id<MTLBuffer> runningVarSqrtWithEps_;
};

#endif /* bn_hpp */
