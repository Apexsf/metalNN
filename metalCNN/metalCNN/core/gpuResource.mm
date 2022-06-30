//
//  gpuResource.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#import "gpuResource.h"


id<MTLBuffer> gpuResource::getBuffer(size_t size) {
    bufferMap_t::iterator iter = bufferMap_.find(size);
    if (iter != bufferMap_.end() && bufferMap_[size].size()) {
        id<MTLBuffer> buffer = bufferMap_[size].back();
        bufferMap_[size].pop_back();
        return buffer;
    } else {
        id <MTLBuffer> buffer = [device_ newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];
        return buffer;
    }
}

void gpuResource::putBuffer(size_t size, id<MTLBuffer> buffer) {
    bufferMap_[size].push_back(buffer);
}

void gpuResource::putBuffer(id<MTLBuffer> buffer) {
    putBuffer(buffer.length / sizeof(float), buffer);
}
