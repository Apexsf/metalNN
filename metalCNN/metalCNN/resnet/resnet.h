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
#include "basicBlock.h"
//#include "metalConstant.metal"




class resnet {
public:
    resnet(std::shared_ptr<gpuResource> resource_);
    
    
private:
    std::shared_ptr<gpuResource> resource_;
};

basicBlock makingBasicBlock(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson);
convParams makingConvParams(NSDictionary* convParamsInfo);
std::map<std::string, tensor> makingConvWeight (NSDictionary* convWeightInfo, convParams params);


uint makingBnParams (NSDictionary* bnParamsInfo);
std::map<std::string, tensor>  makingBnWeight (NSDictionary* bnWeightInfo,
                                               uint params);

#endif /* resnet_h */
