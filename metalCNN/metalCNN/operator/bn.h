//
//  bn.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/21.
//

#ifndef bn_h
#define bn_h
#include "operator.h"

class bn : public op {
public:
    bn(std::shared_ptr<gpuResource> resource, std::string name, uint channel);
    virtual void loadWeight(std::map<std::string, tensor>& weights) override;
    
    void execute (id<MTLBuffer>input, id<MTLBuffer> output);
    
    
private:
    uint channel_;
    id<MTLBuffer> gamma_;
    id<MTLBuffer> beta_;
    id<MTLBuffer> runningMean_;
    id<MTLBuffer> runningVar_;
};

#endif /* bn_hpp */
