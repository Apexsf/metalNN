//
//  resnet.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/27.
//

#include "resnet.h"

resnet::resnet(std::shared_ptr<gpuResource> resource, preLayer& pl,
               basicLayer& bl1, basicLayer& bl2, basicLayer& bl3, basicLayer& bl4):
resource_(resource), preLayer_(pl),
basicLayer1_(bl1),
basicLayer2_(bl2),
basicLayer3_(bl3),
basicLayer4_(bl4){
    
}


id<MTLCommandBuffer> resnet:: forward(const id<MTLBuffer> input, const shape& inShape,
                             id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP){
    
    
    shape preOutShape = preLayer_.getOutputShape(inShape);
    id<MTLBuffer> preOutBuffer = resource_->getBuffer(preOutShape.sizeNC4HW4());
    id<MTLCommandBuffer> commandBuffer = preLayer_.forward(input, inShape, preOutBuffer, commandBufferP);
    
    shape l1b1OutShape = basicLayer1_.block1_.getOutputShape(preOutShape);
    id<MTLBuffer> l1b1OutBuffer = resource_->getBuffer(l1b1OutShape.sizeNC4HW4());
    commandBuffer = basicLayer1_.block1_.forward(preOutBuffer, preOutShape, output, &commandBuffer);
    
    
    
    return commandBuffer;
}
