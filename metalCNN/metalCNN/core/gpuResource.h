//
//  gpuResource.h
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#ifndef gpu_resource_h
#define gpu_resource_h

#import <Metal/Metal.h>
class gpuResource {
public:
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
    
    
    
private:
    id <MTLDevice> device_;
    id <MTLLibrary> library_;
    id <MTLCommandQueue> commandQueue_;
};




#endif






