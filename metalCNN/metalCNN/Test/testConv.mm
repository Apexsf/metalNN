//
//  testConv.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//


#include "testUtils.h"


void testConv(){
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/input.bin";
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/out.bin";
        std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/weight.bin";
        std::string bias_path = "/Users/tinglyfeng/Desktop/metalCNN/script/conv/bias.bin";
        
        shape inShape{2,11,71,83};
        convParams param{5,3,11,9,4,9,4,7};
        convConstant constant = conv::makeConvConstant(inShape, param);
        shape outShape {(uint)constant.out_batch, param.outC, (uint)constant.out_height, (uint)constant.out_width};
        
        conv convOp = conv(resource, "conv", param);
        tensor weight(param.outC, param.inC, param.kernelH, param.kernelW);
        weight.loadFromFile(weight_path.c_str());
        tensor bias(1, param.outC, 1,1);
        bias.loadFromFile(bias_path.c_str());
        std::map<std::string, tensor> weights = {
            {"weight", std::move(weight)},
            {"bias", std::move(bias)}
        };
        convOp.loadWeight(weights);
        
        
        id<MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};
        
        convOp.runOnce(inOutBuffers, &constant);
        
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);
        
        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }

}
