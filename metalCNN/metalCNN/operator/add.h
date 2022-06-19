//
//  add.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#ifndef add_h
#define add_h
#include "operator.h"

class add: public op {
public:
    add(std::shared_ptr<gpuResource> resource, size_t size = 1024 * 5120) : op(resource, std::string("add")) {
        inBuffer1_ = [resource->getDevice() newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];
        inBuffer2_ = [resource->getDevice() newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];
        outBuffer_ = [resource->getDevice() newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];
        
        for(size_t i = 0; i < size; ++i) {
            ((float*)(inBuffer1_.contents))[i] = i;
            ((float*)(inBuffer2_.contents))[i] = 2 * i;
        }
    }
    void execute();
//    virtual void loadWeight(const tensor& t) override {
//
//    }
    float* getOutput() {
        return (float*)outBuffer_.contents;
    }
private:
    id <MTLBuffer> inBuffer1_;
    id <MTLBuffer> inBuffer2_;
    id <MTLBuffer> outBuffer_;
    
};


#endif /* add_hpp */
