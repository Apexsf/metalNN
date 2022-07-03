//
//  postLayer.h
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/3.
//

#ifndef postLayer_h
#define postLayer_h
#include "gpuResource.h"
#include "conv.h"
#include "bn.h"
#include "act.h"
#include "pooling.h"
#include "matmul.h"

class postLayer{
public:
    postLayer(std::shared_ptr<gpuResource> resource, const pooling& avgpooling, const matmul& fc);
    id<MTLCommandBuffer>forward(const id<MTLBuffer> input, const shape& inShape,
                                  id<MTLBuffer> output, id<MTLCommandBuffer>* commandBufferP);
    shape getOutShape(const shape& inShape);
    
  
    
private:
    void makingConstantAndShape(const shape& inShape);
    std::shared_ptr<gpuResource> resource_;
    pooling avgpooling_;
    matmul fc_;
    
    poolingConstant poolingConst_;
    matmulConstant matmulConst_;
};


#endif /* postLayer_h */
