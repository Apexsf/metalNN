//
//  postLayer.c
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#include "postLayer.h"


void postLayer::makingConstantAndShape(const shape& inShape){
    poolingParams avgPoolingPara {inShape.height, inShape.width, 0, 0, 1, 1};
    avgpooling_.resetParams(avgPoolingPara);
    poolingConst_ = pooling::makePoolingConstant(inShape, avgPoolingPara);
    matmulConst_  =  matmulConstant{(int)inShape.batch,poolingConst_.out_slice,(int)fc_.getOutC()};
    
}

postLayer:: postLayer(std::shared_ptr<gpuResource> resource, const  pooling& avgpooling, const matmul& fc) : resource_(resource), avgpooling_(avgpooling), fc_(fc){
    
}

shape postLayer::getOutShape(const shape& inShape){
    return shape{inShape.batch,fc_.getOutC(), 1, 1};
}
//
id<MTLCommandBuffer> postLayer:: forward(const id<MTLBuffer> input, const shape& inShape,
                                         id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP){
    id<MTLCommandBuffer> commandBuffer = commandBufferP? *commandBufferP: [resource_->getCommandQueue() commandBuffer];
    makingConstantAndShape(inShape);
    
    // encode avgpooling
//    id<MTLBuffer> interBuffer1 = resource_->getBuffer(poolingConst_.out_slice * 4 * poolingConst_.out_batch);
    scopeBuffer sb1(resource_, poolingConst_.out_slice * 4 * poolingConst_.out_batch);
    std::vector<id<MTLBuffer>> inOutBuffers {input, sb1.get()};
    avgpooling_.encodeCommand(inOutBuffers, &poolingConst_, commandBuffer);
    
    // encode fc
    inOutBuffers = {sb1.get(), output};
    fc_.encodeCommand(inOutBuffers, &matmulConst_, commandBuffer);
    
//    resource_->putBuffer(sb1.get());
    return commandBuffer;

}
