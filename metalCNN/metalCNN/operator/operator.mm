//
//  operator.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/16.
//

#include "operator.h"

id<MTLBuffer> op:: makingBuffer(size_t byteSize, int mode){
    id<MTLBuffer> buffer = [resource_->getDevice() newBufferWithLength:byteSize options:mode];
    return buffer;
}
