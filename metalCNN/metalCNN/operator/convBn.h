//
//  convBN.h
//  metalCNN
//
//  Created by tinglyfeng on 2022/8/27.
//

#ifndef convBN_h
#define convBN_h

#import <Metal/Metal.h>
#include <memory>
#include "gpuResource.h"
#include "operator.h"
#include "metalConstant.metal"
#include "conv.h"

class convBn: public conv {
public:
    convBn(std::shared_ptr<gpuResource> resource, std::string name, const convParams& params);
    
    virtual void loadWeight(std::map<std::string, tensor>& weights) override;
    
};


#endif /* convBN_h */
