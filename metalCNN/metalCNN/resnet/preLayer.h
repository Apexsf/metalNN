//
//  preLayer.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/2.
//

#ifndef preLayer_h
#define preLayer_h
#include "gpuResource.h"
#include "conv.h"
#include "bn.h"
#include "act.h"
#include "interp.h"
#include "pooling.h"


class preLayer{
public:
    preLayer(std::shared_ptr<gpuResource> resource, const conv& c, const bn& b, const act& a, const pooling& p, const interp& interpBilinear);
    
    id<MTLCommandBuffer>forward(const id<MTLBuffer> input, const shape& inShape,
                                  id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP);
    shape getOutputShape(const shape& inShape);
private:
    std::shared_ptr<gpuResource> resource_;
    void makingConstantAndShape(const shape& inShape);
    
    interp interpBilinear_;
    
    conv conv_;
    bn bn_;
    act relu_;
    pooling maxpooling_;
    
    convConstant convConst_;
    bnConstant bnConst_;
    actConstant actConst_;
    poolingConstant poolingConst_;
    interpBilinearConstant interpConst_;
    
    shape outShape_;
};

#endif /* preLayer_h */
