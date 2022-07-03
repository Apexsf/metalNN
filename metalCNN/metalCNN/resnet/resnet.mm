//
//  resnet.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/27.
//

#include "resnet.h"

shape resnet::getOutputShapeOfLayer(basicLayer& layer, const shape& inShape) {
    shape b1OutputShape = layer.block1_.getOutputShape(inShape);
    shape b2OutputShape = layer.block2_.getOutputShape(b1OutputShape);
    return b2OutputShape;
}

resnet::resnet(std::shared_ptr<gpuResource> resource, preLayer& prel,
               basicLayer& bl1, basicLayer& bl2, basicLayer& bl3, basicLayer& bl4, postLayer& postl):
resource_(resource), preLayer_(prel),
basicLayer1_(bl1),
basicLayer2_(bl2),
basicLayer3_(bl3),
basicLayer4_(bl4),
postLayer_(postl)
{
    
}

id<MTLCommandBuffer> resnet::forwardLayer(const id<MTLBuffer> input, const shape& inShape, id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP, basicLayer& layer){
    
    shape b1OutputShape = layer.block1_.getOutputShape(inShape);
    id<MTLBuffer> b1OutputBuffer = resource_ -> getBuffer(b1OutputShape.sizeNC4HW4());
    id<MTLCommandBuffer> commandBuffer = layer.block1_.forward(input, inShape, b1OutputBuffer, commandBufferP);
    
    commandBuffer = layer.block2_.forward(b1OutputBuffer, b1OutputShape, output, &commandBuffer);
    
    return commandBuffer;
}


id<MTLCommandBuffer> resnet:: forward(const id<MTLBuffer> input, const shape& inShape,
                             id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP){
    
    
    shape preOutShape = preLayer_.getOutputShape(inShape);
    id<MTLBuffer> preOutBuffer = resource_->getBuffer(preOutShape.sizeNC4HW4());
    id<MTLCommandBuffer> commandBuffer = preLayer_.forward(input, inShape, preOutBuffer, commandBufferP);
    
    
    shape layer1OutShape = getOutputShapeOfLayer(basicLayer1_, preOutShape);
    id<MTLBuffer> layer1OutputBuffer = resource_->getBuffer(layer1OutShape.sizeNC4HW4());
    commandBuffer = forwardLayer(preOutBuffer, preOutShape, layer1OutputBuffer, &commandBuffer, basicLayer1_);

    shape layer2OutShape = getOutputShapeOfLayer(basicLayer2_, layer1OutShape);
    id<MTLBuffer> layer2OutputBuffer = resource_->getBuffer(layer2OutShape.sizeNC4HW4());
    commandBuffer  = forwardLayer(layer1OutputBuffer, layer1OutShape, layer2OutputBuffer, &commandBuffer, basicLayer2_);
    
    
    shape layer3Outshape = getOutputShapeOfLayer(basicLayer3_, layer2OutShape);
    id<MTLBuffer> layer3OutputBuffer = resource_->getBuffer(layer3Outshape.sizeNC4HW4());
    commandBuffer = forwardLayer(layer2OutputBuffer, layer2OutShape, layer3OutputBuffer, &commandBuffer, basicLayer3_);
    
    shape layer4Outshape = getOutputShapeOfLayer(basicLayer4_, layer3Outshape);
    id<MTLBuffer> layer4OutputBuffer = resource_->getBuffer(layer4Outshape.sizeNC4HW4());
    commandBuffer = forwardLayer(layer3OutputBuffer, layer3Outshape, layer4OutputBuffer, &commandBuffer, basicLayer4_);
    
    shape postOutShape = postLayer_.getOutShape(layer4Outshape);
    id <MTLBuffer> postOutBuffer = resource_->getBuffer(postOutShape.sizeNC4HW4());
    commandBuffer = postLayer_.forward(layer4OutputBuffer, layer4Outshape, output, &commandBuffer);
    
    
    return commandBuffer;
}
