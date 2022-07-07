//
//  testElemWise.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//

#include "testUtils.h"

void testElemWise() {
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
            std::string input1_path = "/Users/tinglyfeng/Desktop/metalCNN/script/elemWise/input1.bin";
            std::string input2_path = "/Users/tinglyfeng/Desktop/metalCNN/script/elemWise/input2.bin";
            std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/elemWise/out.bin";
        
        shape inShape1{2,11,71,83};
        shape inShape2{2,11,71,83};
        shape outShape{2,11,71,83};
        elemWiseConstant constant{(int)inShape1.batch, (int)divUp(inShape1.channel, 4), (int)inShape1.height, (int)inShape1.width, (int)inShape1.width * (int)inShape1.height};
        elemWise elemWiseOp(resource, "elemWiseMul");

        id<MTLBuffer> inputBuffer1 = makingInputBuffer(resource, input1_path, inShape1);
        id<MTLBuffer> inputBuffer2 = makingInputBuffer(resource, input2_path, inShape2);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer1, inputBuffer2, outputBuffer};

        elemWiseOp.runOnce(inOutBuffers, &constant);
        
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);
        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }

}
