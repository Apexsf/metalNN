//
//  act.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/25.
//

#include "act.h"


act::act(std::shared_ptr<gpuResource> resource, std::string act_name): op(resource, act_name) {
    
}


void execute(id<MTLBuffer> input, id<MTLBuffer> output, const shape& shp) {
    
}
