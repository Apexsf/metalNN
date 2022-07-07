//
//  testResnet.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//

#include "testUtils.h"

void testResNet() {
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        NSString *path = @"/Users/tinglyfeng/Desktop/metalCNN/script/resnet/testData.json";
        NSData *data = [NSData dataWithContentsOfFile:path];
        NSError *error;
        NSDictionary *netInfo = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:&error];
        resnet net = makingResNet(resource, netInfo);
        std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/resnet/input.bin";
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/resnet/out.bin";
        shape inShape {1,3,198,167};
        shape outShape{1,1000,1,1};
    
        id <MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id <MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
    
        id<MTLCommandBuffer> commandBuffer = net.forward(inputBuffer, inShape, outputBuffer, nullptr);
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        tensor metalOutTensor = makingMetalOutTensorNCHW(outputBuffer, outShape);
    
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        diffProfile(torchOutTensor.getRawPointer(), metalOutTensor.getRawPointer(), torchOutTensor.absSize());
    }

}
