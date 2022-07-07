//
//  testAct.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//

#include "testUtils.h"

void testAct(){
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        std::string input_path =       "/Users/tinglyfeng/Desktop/metalCNN/script/act/input.bin";;
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/act/out.bin";
        

        shape inShape{2,11,71,83};
        actConstant constant{(int)inShape.batch, (int)divUp(inShape.channel, 4), (int)inShape.height, (int)inShape.width};
        shape outShape = inShape;
        
        act reluOp(resource, std::string("relu"));

        id<MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};

        reluOp.runOnce(inOutBuffers, &constant);
        
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);
        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }
}
