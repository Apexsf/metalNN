//
//  testBn.m
//  metalCNN
//
//  Created by tinglyfeng on 2022/7/7.
//


#include "testUtils.h"


void testBn() {
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/bn/input.bin";
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/bn/out.bin";
        std::string gamma_path =
            "/Users/tinglyfeng/Desktop/metalCNN/script/bn/gamma.bin";
        std::string beta_path =
            "/Users/tinglyfeng/Desktop/metalCNN/script/bn/beta.bin";
        std::string rm_path =
            "/Users/tinglyfeng/Desktop/metalCNN/script/bn/running_mean.bin";
        std::string rv_path = "/Users/tinglyfeng/Desktop/metalCNN/script/bn/running_var.bin";

        shape inShape {2,64,67,67};
        uint param = inShape.channel;
        bnConstant constant{(int)inShape.batch,(int)divUp(inShape.channel, 4), (int)inShape.height, (int)inShape.width, (int)(inShape.height * inShape.width)};
        shape outShape = inShape;

        bn bnOp = bn(resource, "bn", param);
        tensor gamma(1,inShape.channel,1,1);
        tensor beta(1,inShape.channel,1,1);
        tensor rm(1,inShape.channel,1,1);
        tensor rv(1,inShape.channel,1,1);
        gamma.loadFromFile(gamma_path.c_str());
        beta.loadFromFile(beta_path.c_str());
        rm.loadFromFile(rm_path.c_str());
        rv.loadFromFile(rv_path.c_str());

        std::map<std::string, tensor> bnWeights = {
            {"gamma", std::move(gamma)},
            {"beta", std::move(beta)},
            {"running_mean", std::move(rm)},
            {"running_var", std::move(rv)}
        };
        bnOp.loadWeight(bnWeights);


        id<MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};
        bnOp.runOnce(inOutBuffers, &constant);

        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);

        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }

}

