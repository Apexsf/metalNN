//
//  testInterp.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//


#include "testUtils.h"

void testInterp(){
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        std::string input_path =       "/Users/tinglyfeng/Desktop/metalCNN/script/interp/input.bin";
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/interp/out.bin";

        shape inShape{3,29,37,67};
        interpParams param {interpParams::targetMode::SIZE , 159 ,229, 0.0, 0.0};
        interpBilinearConstant constant = interp::makingBilinearConstant(inShape, param);
        shape outShape{(uint)constant.out_batch, inShape.channel, (uint)constant.out_height, (uint)constant.out_width};
        
        interp interpOp(resource, std::string("interpBilinear"), param);

        id<MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};

        interpOp.runOnce(inOutBuffers, &constant);
        
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);
        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }

}
