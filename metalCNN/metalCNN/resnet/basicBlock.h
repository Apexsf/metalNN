//
//  basicBlock.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/30.
//

#ifndef basicBlock_h
#define basicBlock_h

#include "gpuResource.h"
#include "conv.h"
#include "bn.h"
#include "act.h"
#include "elemWise.h"

class basicBlock {
public:
    
    basicBlock(std::shared_ptr<gpuResource> resource, convParams& convPara1,
               convParams& convPara2, uint bnChannel1, uint bnChannel2);
    
    basicBlock(std::shared_ptr<gpuResource> resource, conv& conv1,
               conv& conv2, bn& bn1, bn& bn2);
//    basicBlock(std::shared_ptr<gpuResource> resource, NSDictionary* blockInfo);
    void loadWeights(std::map<std::string, tensor>& convWeight1, std::map<std::string, tensor>& convWeight2, std::map<std::string, tensor>& bnWeights1, std::map<std::string, tensor>& bnWeights2);
    
    void setDownSampleModule(convParams& convPara3, uint bnChannel3,
                             std::map<std::string, tensor>& convWeight3,
                             std::map<std::string, tensor>& bnWeight3);
    
    void setDownSampleModule(conv& conv3, bn& bn3);
    
//    basicBlock(NSDictionary *infoFromJson);
    
    id<MTLCommandBuffer> forward(const id<MTLBuffer> input,
                                 const shape& inShape,
                                 id <MTLBuffer> output,
                                 id<MTLCommandBuffer>*
                                 commandBufferP);
    
    ~basicBlock();
private:
    void cleanDownSampleModule();
    void makingConstantAndShape(const shape& inShape);
    std::shared_ptr<gpuResource> resource_;
    conv conv1_;
    conv conv2_;
    bn bn1_;
    bn bn2_;
    act relu_;
    elemWise add_;
    
    bool hasDownSample_ = false;
    conv* conv3P_ = nullptr; // perform downsample if exists
    bn* bn3P_ = nullptr; // perform downsample if exists
    convConstant* convConst3_ = nullptr; // perform downsample if exists
    bnConstant* bnConst3_ = nullptr;
    
    convConstant convConst1_;
    convConstant convConst2_;
    actConstant actConst1_;
    actConstant actConst2_;
    bnConstant bnConst1_;
    bnConstant bnConst2_;
    elemWiseConstant addConst_;
    
    shape outShape1_;  // output shape from conv1
    shape outShape2_; // output shape from conv2
};


#endif /* basicBlock_h */
