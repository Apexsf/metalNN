//
//  preLayer.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/2.
//

#include "preLayer.h"

preLayer::preLayer(std::shared_ptr<gpuResource> resource, const conv& c, const bn& b, const act& a, const pooling& p, const interp& interpBilinear): resource_(resource), conv_(c), bn_(b), relu_(a), maxpooling_(p), interpBilinear_(interpBilinear){
    
}

shape preLayer::getOutputShape(const shape& inShape){
    interpBilinearConstant interpConst = interp::makingBilinearConstant(inShape, interpBilinear_.getParams());
    shape interpShape{(uint)interpConst.out_batch, inShape.channel, (uint)interpConst.out_height,
       (uint)interpConst.out_width};
    convConstant convConst = conv::makeConvConstant(interpShape, conv_.getParams());
    shape outShape1 {(uint)convConst.out_batch, conv_.getParams().outC, (uint)convConst.out_height, (uint)convConst.out_width};
    poolingConstant poolingConst = pooling::makePoolingConstant(outShape1, maxpooling_.getParams());
    shape outshape2 {(uint) poolingConst.out_batch , (uint) conv_.getParams().outC,
        (uint)poolingConst.out_height, (uint) poolingConst.out_width
    };
    return outshape2;
    
}

void preLayer::makingConstantAndShape(const shape& inShape){
    interpConst_ = interp::makingBilinearConstant(inShape, interpBilinear_.getParams());
    shape interpShape{(uint)interpConst_.out_batch, inShape.channel, (uint)interpConst_.out_height,
       (uint)interpConst_.out_width};
    convConst_   = conv::makeConvConstant(interpShape, conv_.getParams());
    bnConst_ = bnConstant{convConst_.out_batch, convConst_.out_slice, convConst_.out_height, convConst_.out_width, convConst_.out_size};
    actConst_ = actConstant{convConst_ .out_batch, convConst_.out_slice, convConst_.out_height,convConst_.out_width};
    poolingConst_ = pooling::makePoolingConstant({(uint)convConst_.out_batch, (uint)conv_.getParams().outC, (uint)convConst_.out_height, (uint)convConst_.out_width}, maxpooling_.getParams());
    outShape_ = shape{(uint)poolingConst_.out_batch, conv_.getParams().outC, (uint)poolingConst_.out_height, (uint)poolingConst_.out_width};
}

id<MTLCommandBuffer> preLayer:: forward (const id<MTLBuffer> input, const shape& inShape,
                              id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP) {
    id<MTLCommandBuffer> commandBuffer = commandBufferP? *commandBufferP: [resource_->getCommandQueue() commandBuffer];
    
    makingConstantAndShape(inShape);
    

//    id<MTLBuffer> interBuffer1 = resource_->getBuffer(convConst_.out_batch * convConst_.out_slice * 4 * convConst_.out_size);
    scopeBuffer sb1(resource_, interpConst_.out_batch * interpConst_.out_slice * 4
                    * interpConst_.out_size);
    std::vector<id<MTLBuffer>> inOutBuffers {input, sb1.get()};
    interpBilinear_.encodeCommand(inOutBuffers, &interpConst_, commandBuffer);
    
    // encode conv
    scopeBuffer sb2(resource_, convConst_.out_batch * convConst_.out_slice * 4 * convConst_.out_size);
    
   inOutBuffers = {sb1.get(), sb2.get()};
    conv_.encodeCommand(inOutBuffers, &convConst_, commandBuffer);
    
//     encode bn1
    inOutBuffers = {sb2.get(), sb2.get()};
    bn_.encodeCommand(inOutBuffers, &bnConst_, commandBuffer);
    
    
// encode relu
    inOutBuffers = {sb2.get(), sb2.get()};
    relu_.encodeCommand(inOutBuffers, &actConst_, commandBuffer);
//
    inOutBuffers = {sb2.get(), output};
    maxpooling_.encodeCommand(inOutBuffers, &poolingConst_, commandBuffer);
    
//    resource_->putBuffer(interBuffer1);
//
    return commandBuffer;
    
}
