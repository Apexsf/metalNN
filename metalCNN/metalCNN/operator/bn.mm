//
//  bn.cpp
//  metalCNN
//
//  Created by tinglyfeng on 2022/6/21.
//

#include "bn.h"

bn::bn(std::shared_ptr<gpuResource> resource, std::string name, uint channel) :  op(resource, name) , channel_(channel){

}

void bn::loadWeight(std::map<std::string, tensor>& weights){
    
}
