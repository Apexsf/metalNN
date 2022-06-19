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

struct convRunTimeConstant{
    uint in_batch;
    uint in_slice; // slice per batch
    uint in_size; // size per slice, h*w
    uint in_height;
    uint in_width;
    
    uint out_batch;
    uint out_slice; // slice per batch
    uint out_size; //size per slice, h*w
    uint out_height;
    uint out_width;
    
    uint kernel_h;
    uint kernel_w;
    uint kernel_size;
    uint stride_h;
    uint stride_w;
    uint pad_x;
    uint pad_y;
};

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
    
    void execute(id<MTLBuffer> input, id<MTLBuffer> output, const convRunTimeConstant& constant);
    virtual void loadWeight(const tensor& t) override ;
    
private:
    id<MTLBuffer> weight_;
    id<MTLBuffer> bias_; // todo: taking bias into consideration
    convParams params_;
};

#endif /* conv_hpp */
