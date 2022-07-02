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
//#include "metalConstant.metal"




struct basicLayer {
    basicBlock block1_;
    basicBlock block2_;
};


class resnet {
public:
    resnet(std::shared_ptr<gpuResource> resource_);
    
    
private:
    std::shared_ptr<gpuResource> resource_;
//    preLayer preLayer_;
    basicLayer basicLayer1_;
    basicLayer basicLayer2_;
    basicLayer basicLayer3_;
    basicLayer basicLayer4_;
    
};



#endif /* resnet_h */
