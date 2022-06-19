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
    uint padY;
    uint padX;
    uint strideH;
    uint strideW;
};

class conv : public op{
public:
    conv(std::shared_ptr<gpuResource> resource, std::string name, const convParams& params);
    const convParams& getParams () const {
        return params_;
    }
    
    void execute(id<MTLBuffer> input, id<MTLBuffer> output, const convConstant& constant);
    virtual void loadWeight(const tensor& t) override ;
    
private:
    id<MTLBuffer> weight_;
    id<MTLBuffer> bias_; // todo: taking bias into consideration
    convParams params_;
};

#endif /* conv_hpp */
