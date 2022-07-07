//
//  utils.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//

#import <Foundation/Foundation.h>
#include <string>
#include <Metal/Metal.h>
#include "tensor.h"
#include "gpuResource.h"


id<MTLBuffer> makingInputBuffer(std::shared_ptr<gpuResource>resource, std::string input_path, const shape& shp){
    tensor input_tensor(shp);
    input_tensor.loadFromFile(input_path.c_str());
    input_tensor.reInterpret(tensor::interpOrder::NC4HW4);
    id<MTLBuffer> input_buffer = [resource->getDevice()
                                  newBufferWithLength:input_tensor.memSize() * sizeof(float) options:MTLResourceStorageModeShared];
    memcpy(input_buffer.contents, input_tensor.getRawPointer(), input_tensor.memSize() * sizeof(float));
    return input_buffer;
}

tensor makingTorchOutTensorNCHW(std::string output_path, const shape& shp) {
    tensor torchOutTensor(shp);
    torchOutTensor.loadFromFile(output_path.c_str());
    return torchOutTensor;
}

tensor makingMetalOutTensorNCHW(id<MTLBuffer> outBuffer, const shape& shp){
    tensor metalOutTensor(shp);
    metalOutTensor.loadFromMemory((float*)outBuffer.contents, tensor::interpOrder::NC4HW4   );
    metalOutTensor.reInterpret(tensor::interpOrder::NCHW);
    return metalOutTensor;
}

void diffProfile(float* p1, float* p2, size_t size){

    float diff;
    float total_diff = 0;
    float max_diff = 0;
    size_t diff_cnt = 0;
    for(size_t i = 0; i < size; ++i){
        diff = std::abs( (p1[i] - p2[i]) );
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
        if (p1[i] != p2[i]){
            diff_cnt+=1;
        }
        if (total_diff > 1){
            std::cout;
        }
    }
    std::cout << "total diff : " << total_diff << std::endl;
    std::cout << "max diff : " << max_diff << std::endl;
}
