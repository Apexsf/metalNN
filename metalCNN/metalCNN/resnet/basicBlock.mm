//
//  basicBlock.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/30.
//

#include "basicBlock.h"

basicBlock::basicBlock(std::shared_ptr<gpuResource> resource, convParams& convPara1,
                       convParams& convPara2, uint bnChannel1, uint bnChannel2):resource_(resource),
conv1_(resource, "conv", convPara1),
conv2_(resource, "conv", convPara2), relu_(resource, "relu"), bn1_(resource, "bn", bnChannel1), bn2_(resource, "bn", bnChannel2), add_(resource, "elemWiseAdd"){
    
}

basicBlock:: basicBlock(std::shared_ptr<gpuResource> resource, conv& conv1,
                        conv& conv2, bn& bn1, bn& bn2):resource_(resource), conv1_(conv1), conv2_(conv2), bn1_(bn1), bn2_(bn2), relu_(resource, "relu"), add_(resource, "elemWiseAdd") {
    
}



basicBlock::~basicBlock(){

}

shape basicBlock::getOutputShape(const shape& inShape){
    
    convConstant convConst1 = conv::makeConvConstant(inShape, conv1_.getParams());
    shape outShape1 = shape{(uint)convConst1.out_batch, conv1_.getParams().outC, (uint)convConst1.out_height, (uint)convConst1.out_width};
    convConstant convConst2 = conv::makeConvConstant(outShape1, conv2_.getParams());
    shape outShape2 {(uint)convConst2.out_batch, (uint)conv2_.getParams().outC, (uint)convConst2.out_height, (uint)convConst2.out_width};
    return outShape2;
}

//basicBlock::basicBlock(std::shared_ptr<gpuResource> resource, NSDictionary* blockInfo){
//
//}


void basicBlock::loadWeights(std::map<std::string, tensor>& convWeight1, std::map<std::string, tensor>& convWeight2, std::map<std::string, tensor>& bnWeights1, std::map<std::string, tensor>& bnWeights2) {
    conv1_.loadWeight(convWeight1);
    conv2_.loadWeight(convWeight2);
    bn1_.loadWeight(bnWeights1);
    bn2_.loadWeight(bnWeights2);
}



void basicBlock::makingConstantAndShape(const shape& inShape) {
    convConst1_ = conv::makeConvConstant(inShape, conv1_.getParams());
    outShape1_ = shape{(uint)convConst1_.out_batch, conv1_.getParams().outC, (uint)convConst1_.out_height, (uint)convConst1_.out_width};
    bnConst1_ = bnConstant{convConst1_.out_batch, convConst1_.out_slice, convConst1_.out_height, convConst1_.out_width, convConst1_.out_size};
    
    actConst1_  = actConstant{convConst1_.out_batch, convConst1_.out_slice,
        convConst1_.out_height, convConst1_.out_width
    };
    
    convConst2_ = conv::makeConvConstant(outShape1_, conv2_.getParams());
    bnConst2_ = bnConstant {convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width, convConst2_.out_size};
    
    outShape1_ = shape{(uint)convConst1_.out_batch, (uint)conv1_.getParams().outC, (uint)convConst1_.out_height, (uint)convConst1_.out_width};
    outShape2_ = shape {(uint)convConst2_.out_batch, (uint)conv2_.getParams().outC, (uint)convConst2_.out_height, (uint)convConst2_.out_width};
    
    addConst_ = elemWiseConstant {convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width, convConst2_.out_size};
    
    actConst2_ = actConstant{convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width};
    
    if (hasDownSample_) {
        convConst3_ = convConstant( conv::makeConvConstant(inShape, conv3_->getParams()) );
        bnConst3_ = bnConstant{convConst2_.out_batch, convConst2_.out_slice, convConst2_.out_height, convConst2_.out_width, convConst2_.out_size};
    }
}


id<MTLCommandBuffer> basicBlock::forward(const id<MTLBuffer> input,
                                         const shape& inShape,
                                         id <MTLBuffer> output,
                                         id<MTLCommandBuffer>* commandBufferP) {
    
    id<MTLCommandBuffer> commandBuffer = commandBufferP? *commandBufferP: [resource_->getCommandQueue() commandBuffer];
    
    makingConstantAndShape(inShape);
    
    
    // encode conv1
//    id<MTLBuffer> interBuffer1 = resource_->getBuffer(convConst1_.out_batch * convConst1_.out_slice * 4 * convConst1_.out_size);
    scopeBuffer sb1(resource_, convConst1_.out_batch * convConst1_.out_slice * 4 * convConst1_.out_size);
    std::vector<id<MTLBuffer>> inOutBuffers {input, sb1.get()};
    conv1_.encodeCommand(inOutBuffers, &convConst1_, commandBuffer);
    
    
    // encode bn1
    inOutBuffers = {sb1.get(), sb1.get()};
    bn1_.encodeCommand(inOutBuffers, &bnConst1_, commandBuffer);

 
    
    //encode relu
    inOutBuffers = {sb1.get(), sb1.get()};
    relu_.encodeCommand(inOutBuffers, &actConst1_, commandBuffer);

    
    // encode conv2
//    id<MTLBuffer> interBuffer2 = resource_->getBuffer(convConst2_.out_batch * convConst2_.out_slice * 4 * convConst2_.out_size);
    scopeBuffer sb2 (resource_, convConst2_.out_batch * convConst2_.out_slice * 4 * convConst2_.out_size);
    inOutBuffers = {sb1.get(), sb2.get()};
    conv2_.encodeCommand(inOutBuffers, &convConst2_, commandBuffer);

    
    // encode bn2
    inOutBuffers = {sb2.get(), sb2.get()};
    bn2_.encodeCommand(inOutBuffers, &bnConst2_, commandBuffer);
    
    if(hasDownSample_){
//        id<MTLBuffer> interBuffer3 = resource_->getBuffer(outShape2_.sizeNC4HW4());
        scopeBuffer sb3 (resource_, outShape2_.sizeNC4HW4());
        
        inOutBuffers = {input, sb3.get()};
        conv3_->encodeCommand(inOutBuffers, &convConst3_, commandBuffer);
        inOutBuffers = {sb3.get(), sb3.get()};
        bn3_->encodeCommand(inOutBuffers, &bnConst3_, commandBuffer);
        inOutBuffers = {sb3.get(), sb2.get(), output};
        add_.encodeCommand(inOutBuffers, &addConst_, commandBuffer);
//        resource_->putBuffer(sb3.get());
    } else {
        // encode add
        inOutBuffers = {input, sb2.get(), output};
        add_.encodeCommand(inOutBuffers, &addConst_, commandBuffer);
    }

//    // encode relu
    inOutBuffers = {output, output};
    relu_.encodeCommand(inOutBuffers, &actConst2_, commandBuffer);
    
//    resource_->putBuffer(interBuffer1);
//    resource_->putBuffer(interBuffer2);
    
    
    return commandBuffer;
    
    
}

void basicBlock::setDownSampleModule(convParams& convPara3, uint bnChannel3,
                         std::map<std::string, tensor>& convWeight3,
                                     std::map<std::string, tensor>& bnWeight3) {
    if (hasDownSample_) return;
    hasDownSample_ = true;
    conv3_ = std::make_shared<conv>(resource_, "conv", convPara3);
    bn3_ = std::make_shared<bn>(resource_, "bn", bnChannel3);
}

void basicBlock:: setDownSampleModule(conv& conv3, bn& bn3){
    if(hasDownSample_) return;
    hasDownSample_ = true;
    conv3_ = std::shared_ptr<conv>(new conv(conv3));
    bn3_ = std::shared_ptr<bn>(new bn(bn3));
}
