//
//  testPooling.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//

#include "testUtils.h"

void testPooling(){
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/pooling/input.bin";
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/pooling/out.bin";
        
        shape inShape{2,11,71,83};
        poolingParams param {9,7,4,3,4,7};
        poolingConstant constant = pooling::makePoolingConstant(inShape, param);
        shape outShape {(uint)constant.out_batch, inShape.channel, (uint)constant.out_height, (uint)constant.out_width};
        
        pooling poolingOp(resource, "poolingAvg", param);
        
        id<MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};
        
        poolingOp.runOnce(inOutBuffers, &constant);
        
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);
        
        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }
}
