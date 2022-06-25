//
//  operator.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#ifndef operator_h
#define operator_h
#import <Metal/Metal.h>

#include <string>
#include <memory>
#include <map>
#include <cmath>
#include <vector>
#include "gpuResource.h"
#include "tensor.h"

class op {
public:
    op(std::shared_ptr<gpuResource> resource, std::string name): resource_(resource)
    , opName_(name){
        shader_ = [resource->getLibrary() newFunctionWithName:@(name.c_str())];
        pso_ = [resource->getDevice() newComputePipelineStateWithFunction:shader_ error:NULL];
    }
    
    std::shared_ptr<gpuResource>  getResource (){
        return resource_;
    }
    
    id<MTLBuffer> makingBuffer(size_t byteSize, int mode);
    
    const id <MTLFunction>& getShader() const  {
        return shader_;
    }
    
    const id <MTLComputePipelineState> getPSO() const {
        return pso_;
    }
    
    id<MTLCommandBuffer> newCommandBuffer(){
        return [resource_->getCommandQueue() commandBuffer];
    }
    

    virtual void setBuffer (std::vector<id<MTLBuffer>>& inOutBuffers, id<MTLComputeCommandEncoder> commandEncoder) = 0;
    virtual void setConstant(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) = 0;
    virtual void dispatch(void* constantP, id<MTLComputeCommandEncoder> commandEncoder) = 0;
    
    void run(std::vector<id<MTLBuffer>>& inOutBuffers, void* constantP);
    
    virtual void loadWeight(std::map<std::string, tensor>& weights) = 0;
private:
    std::string opName_;
    std::shared_ptr<gpuResource> resource_;
    id <MTLFunction> shader_;
    id <MTLComputePipelineState> pso_;
};

#endif /* operator_hpp */
