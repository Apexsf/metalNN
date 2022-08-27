//
//  testConvBn.h
//  metalCNN
//
//  Created by tinglyfeng on 2022/8/27.
//

#ifndef testConvBn_h
#define testConvBn_h


#include "testUtils.h"

void testConvBn(){
    @autoreleasepool {
        std::shared_ptr<gpuResource> resource = std::make_shared<gpuResource>();
        std::string input_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/input.bin";
        std::string output_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/out.bin";
        std::string weight_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/weight.bin";
        std::string bias_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/bias.bin";
        
        std::string gamma_path =
            "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/gamma.bin";
        std::string beta_path =
            "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/beta.bin";
        std::string rm_path =
            "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/running_mean.bin";
        std::string rv_path = "/Users/tinglyfeng/Desktop/metalCNN/script/convBn/running_var.bin";
        
        
        shape inShape{2,11,71,83};
        convParams param{5,3,11,9,4,9,4,7};
        convConstant constant = conv::makeConvConstant(inShape, param);
        shape outShape {(uint)constant.out_batch, param.outC, (uint)constant.out_height, (uint)constant.out_width};
        
        convBn convBnOp = convBn(resource, "conv", param);
        
        tensor weight(param.outC, param.inC, param.kernelH, param.kernelW);
        tensor bias(1, param.outC, 1,1);
        weight.loadFromFile(weight_path.c_str());
        bias.loadFromFile(bias_path.c_str());
        
        tensor gamma(1,inShape.channel,1,1);
        tensor beta(1,inShape.channel,1,1);
        tensor rm(1,inShape.channel,1,1);
        tensor rv(1,inShape.channel,1,1);
        gamma.loadFromFile(gamma_path.c_str());
        beta.loadFromFile(beta_path.c_str());
        rm.loadFromFile(rm_path.c_str());
        rv.loadFromFile(rv_path.c_str());
        
        std::map<std::string, tensor> weights = {
            {"weight", std::move(weight)},
            {"bias", std::move(bias)},
            {"gamma", std::move(gamma)},
            {"beta", std::move(beta)},
            {"running_mean", std::move(rm)},
            {"running_var", std::move(rv)},
        };
        
        convBnOp.loadWeight(weights);
        
        
        id<MTLBuffer> inputBuffer = makingInputBuffer(resource, input_path, inShape);
        id<MTLBuffer> outputBuffer = resource->getBuffer(outShape.sizeNC4HW4());
        std::vector<id<MTLBuffer>> inOutBuffers{inputBuffer, outputBuffer};
        
        convBnOp.runOnce(inOutBuffers, &constant);
        
        tensor torchOutTensor = makingTorchOutTensorNCHW(output_path, outShape);
        tensor metalOutput = makingMetalOutTensorNCHW(outputBuffer, outShape);
        
        diffProfile(torchOutTensor.getRawPointer(), metalOutput.getRawPointer(), torchOutTensor.absSize());
    }
}


#endif /* testConvBn_h */

