//
//  gpuResource.h
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#ifndef gpu_resource_h
#define gpu_resource_h

#import <Metal/Metal.h>
#include <map>
#include <vector>
class gpuResource {
public:
    using  bufferVec_t = std::vector<id<MTLBuffer>> ;
    using bufferMap_t = std::map<size_t, bufferVec_t>;
    
    gpuResource(){
        device_ = MTLCreateSystemDefaultDevice();
        library_ = [device_ newDefaultLibrary];
        commandQueue_ = [device_ newCommandQueue];
    }
    
    const id<MTLDevice>& getDevice () const {
        return device_;
    }
    
    const id<MTLLibrary>& getLibrary () const {
        return library_;
    }
    
    const id<MTLCommandQueue>& getCommandQueue() const {
        return commandQueue_;
    }
    
    id<MTLBuffer> getBuffer (size_t );
    void putBuffer(id<MTLBuffer>);
    void putBuffer(size_t, id<MTLBuffer>);
    
    
    
private:

    bufferMap_t bufferMap_;
    id <MTLDevice> device_;
    id <MTLLibrary> library_;
    id <MTLCommandQueue> commandQueue_;
};




#endif






