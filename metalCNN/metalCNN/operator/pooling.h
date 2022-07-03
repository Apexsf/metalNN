//
//  pooling.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#ifndef pooling_h
#define pooling_h

#include "operator.h"
#include "metalConstant.metal"


struct poolingParams {
    uint kernelH;
    uint kernelW;
    uint padH;
    uint padW;
    uint strideH;
    uint strideW;
};


class pooling:public op {
public:
    pooling (std::shared_ptr<gpuResource> resource, std::string poolingMode,
            const poolingParams& params);
    
    static poolingConstant makePoolingConstant(const shape& inShape, const poolingParams& params);
    
    poolingParams getParams() const {
        return params_;
    }
    void resetParams (const poolingParams& params) {
        params_ = params;
    }
    
    virtual void loadWeight(std::map<std::string, tensor>& weights) override {
        
    }
    
    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) override;
    
private:
    std::string poolingMode_;
    poolingParams params_;
};




#endif /* pooling_hpp */
