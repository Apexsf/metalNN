//
//  resnet.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/27.
//

#ifndef resnet_h
#define resnet_h

#include "gpuResource.h"
#include "conv.h"
#include "bn.h"
#include "act.h"
#include "elemWise.h"
//#include "metalConstant.metal"

class basicBlock {
public:
    
    basicBlock(std::shared_ptr<gpuResource> resource, convParams& convPara1,
               convParams& convPara2, uint bnChannel1, uint bnChannel2);
    void loadWeights(std::map<std::string, tensor>& convWeight1, std::map<std::string, tensor>& convWeight2, std::map<std::string, tensor>& bnWeights1, std::map<std::string, tensor>& bnWeights2);
    
    id<MTLCommandBuffer> forward(id<MTLBuffer> input,shape& inShape,
                                 id<MTLCommandBuffer>*
                                 commandBufferP);
    

private:
    void makingConstantAndShape(shape& inShape);
    std::shared_ptr<gpuResource> resource_;
    conv conv1_;
    bn bn1_;
    act relu_;
    conv conv2_;
    bn bn2_;
    
    convConstant convConst1_;
    convConstant convConst2_;
    actConstant actConst_;
    bnConstant bnConst1_;
    bnConstant bnConst2_;
    
    shape outShape1_;  // output shape from conv1
    shape outShape2_; // output shape from conv2
};


class resnet {
public:
    resnet(std::shared_ptr<gpuResource> resource_);
    
    
private:
    std::shared_ptr<gpuResource> resource_;
};

#endif /* resnet_h */