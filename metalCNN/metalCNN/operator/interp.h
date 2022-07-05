//
//  interp.h
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#ifndef interp_h
#define interp_h

#include "operator.h"
#include "metalConstant.metal"


// if outH and outW are not 0,
struct interpParams {
    enum class targetMode{
        SIZE = 0,
        SCALE
    };
    targetMode mode;
    //target output mode
    uint outH ;
    uint outW ;
    // scale mode
    float scaleH;
    float scaleW;

};


class interp : public op {
public:
    enum class interpMode {
        NEAREST = 0,
        BILINEAR
    };
    
    interp(std::shared_ptr<gpuResource> resource, std::string interpModeName,
           const interpParams& params);
    
    static interpBilinearConstant makingBilinearConstant(const shape& inShape, const interpParams& params);
    
    
    virtual void loadWeight(std::map<std::string, tensor>& weights) override {
        
    }
    
    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) override;
    
    
private:
    interpMode  mode_ = interpMode::BILINEAR;
    interpParams params_;
};








#endif /* interp_h */
