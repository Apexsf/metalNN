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
    void loadWeights(std::map<std::string, tensor>& convWeight1, std::map<std::string, tensor>& convWeight2, std::map<std::string, tensor>& bnWeights1, std::map<std::string, tensor>& bnWeights2);
    
//    basicBlock(NSDictionary *infoFromJson);
    
    id<MTLCommandBuffer> forward(const id<MTLBuffer> input,
                                 const shape& inShape,
                                 id <MTLBuffer> output,
                                 id<MTLCommandBuffer>*
                                 commandBufferP);
    
private:
    void makingConstantAndShape(const shape& inShape);
    std::shared_ptr<gpuResource> resource_;
    conv conv1_;
    bn bn1_;
    act relu_;
    conv conv2_;
    bn bn2_;
    
    elemWise add_;
    
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
