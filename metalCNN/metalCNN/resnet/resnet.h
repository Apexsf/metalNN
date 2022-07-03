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
#include "pooling.h"
#include "basicBlock.h"
#include "preLayer.h"
//#include "metalConstant.metal"




struct basicLayer {
    basicBlock block1_;
    basicBlock block2_;
};


class resnet {
public:
    resnet(std::shared_ptr<gpuResource> resource, preLayer& pl,
           basicLayer& bl1, basicLayer& bl2, basicLayer& bl3, basicLayer& bl4);
    
    id<MTLCommandBuffer>forward(const id<MTLBuffer> input, const shape& inShape,
                                  id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP);
    
private:
    shape getOutputShapeOfLayer(basicLayer& layer, const shape& inShape);
    id<MTLCommandBuffer> forwardLayer(const id<MTLBuffer> input, const shape& inShape, id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP,basicLayer& layer);
    
    std::shared_ptr<gpuResource> resource_;
    preLayer preLayer_;
    basicLayer basicLayer1_;
    basicLayer basicLayer2_;
    basicLayer basicLayer3_;
    basicLayer basicLayer4_;
    
};



#endif /* resnet_h */
