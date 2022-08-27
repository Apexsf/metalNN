//
//  conv.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#ifndef conv_h
#define conv_h


#import <Metal/Metal.h>
#include <memory>
#include "gpuResource.h"
#include "operator.h"
#include "metalConstant.metal"


struct convParams{
    uint kernelH;
    uint kernelW;
    uint inC;
    uint outC;
    uint padH;
    uint padW;
    uint strideH;
    uint strideW;
};


class conv : public op{
public:
    static convConstant makeConvConstant(const shape& inShape, const convParams& param);
    
    static shape calOutShape(const shape& inShape, const convParams& params);
    
    conv(std::shared_ptr<gpuResource> resource, std::string name, const convParams& params);
    const convParams& getParams () const {
        return params_;
    }
    const id <MTLBuffer> getWeight() const {
        return weight_;
    }
    const id <MTLBuffer> getBias() const {
        return bias_;
    }
 
//    void execute(id<MTLBuffer> input, id<MTLBuffer> output, const convConstant& constant);
    virtual void loadWeight(std::map<std::string, tensor>& weights) override;
    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void setConstant(void* constantP,  id<MTLComputeCommandEncoder> commandEncoder) override;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) override;
protected:
    id<MTLBuffer> weight_;
    id<MTLBuffer> bias_; // todo: taking bias into consideration
    convParams params_;
};

#endif /* conv_hpp */
