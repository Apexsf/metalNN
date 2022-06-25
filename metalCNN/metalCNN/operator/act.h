//
//  act.hpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#ifndef act_h
#define act_h
#include "operator.h"

class act : public op {
    act(std::shared_ptr<gpuResource> resource, std::string act_name);
    void execute(id <MTLBuffer> input, id<MTLBuffer> output, const shape& shp);

};


#endif /* act_hpp */
