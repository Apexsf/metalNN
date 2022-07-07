//
//  testMatMul.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//

#include "testUtils.h"

void testMatMul() {
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/input.bin";
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/out.bin";
        std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/weight.bin";
        std::string bias_path = "/Users/tinglyfeng/Desktop/metalCNN/script/matmul/bias.bin";

        shape inShape{1,512,1,1};
        shape outShape {1,1000,1,1};
        matmulConstant constant {1, (int)divUp(inShape.channel, 4), (int)outShape.channel};

        matmul mmOp(resource, inShape.channel, outShape.channel);
        tensor weight(1,1,outShape.channel, inShape.channel);
        tensor bias(1,outShape.channel,1,1);
        weight.loadFromFile(weight_path.c_str());
        bias.loadFromFile(bias_path.c_str());
        std::map<std::string, tensor> weights = {
            {"weight", std::move(weight)},
            {"bias", std::move(bias)}
        };
        mmOp.loadWeight(weights);
        
        id<MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};

        mmOp.runOnce(inOutBuffers, &constant);
        
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);
        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }
}
