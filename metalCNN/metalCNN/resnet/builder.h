//
//  builder.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/2.
//

#ifndef builder_h
#define builder_h
#include "basicBlock.h"
#include "resnet.h"
#include "preLayer.h"

resnet makingResNet(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson);

basicBlock makingBasicBlock(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson);

basicLayer makingBasicLayer(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson);

conv makingConv(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson);

bn makingBN(std::shared_ptr<gpuResource> resource ,NSDictionary *infoFromJson);

preLayer makingPreLayer(std::shared_ptr<gpuResource> resource, NSDictionary* infoFromJson);
basicLayer makingBasicLayer(std::shared_ptr<gpuResource> resource, NSDictionary* infoFromJson);
                            

poolingParams makingPoolingParams(NSDictionary* poolingParamsInfo);
pooling makingPooling(std::shared_ptr<gpuResource>resource, std::string poolingMode, NSDictionary* infoFromJson);

convParams makingConvParams(NSDictionary* convParamsInfo);
std::map<std::string, tensor> makingConvWeight (NSDictionary* convWeightInfo, convParams params);
uint makingBnParams (NSDictionary* bnParamsInfo);
std::map<std::string, tensor>  makingBnWeight (NSDictionary* bnWeightInfo,
                                               uint params);



#endif /* builder_h*/
